#!/usr/bin/env python3
"""色仕分け課題: CLIPの画像・文章照合 → Pandaの摩擦把持 → MP4録画。

Python 3.11を推奨。最初に requirements.txt を導入する。
  python run_sorting.py --output output
  python run_sorting.py --no-video --output smoke_test

重要な限定:
* CLIPは自由回答型ではなく、候補文との類似度で色を分類する画像・言語モデル。
* 候補に未知色purpleを含め、予測が未知色または候補間差が小さい場合は仕分けを保留する。
* 位置推定は真上カメラの色付き塊を使う古典的な画像処理。深層学習の物体検出ではない。
* 位置推定は失敗後の再接近先の決定だけに使い、初回の供給位置は既知治具のまま。
* 関節は逆運動学と位置制御。把持は接触・摩擦のみで、物体の瞬間移動や固定拘束なし。
* 動画の時刻はシミュレーション時間。推論待ちなどの実時間とは異なる。
* 本コードは学習用シミュレーション専用。実機に接続しない。
* 把持検知・再試行の安全判定にはシミュレータの接触・高さ・位置を使う。
* --fault は比較試験用の明示的な故障注入。回復保証ではない。
* すべり検知: 保持中の高さ低下を監視し、把持力を1.6倍ずつ（上限2.5倍）強める。回復保証ではない。
* --multi-view: 俯瞰に加えて真上ミニ窓と手先追従カメラを合成する録画モード。
* --wall: 供給列と回収皿の間に低い壁を置き、経由点（壁上空）を挟む搬送に切り替える。
* 統計付き自動試験: python test_suite.py --output stats_suite
"""
from __future__ import annotations
import argparse
from argparse import Namespace
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = 'openai/clip-vit-base-patch32'
COLORS = ['red', 'green', 'blue']
UNKNOWN = ['purple']
CANDIDATES = COLORS + UNKNOWN
JP = {'red': '赤', 'green': '緑', 'blue': '青', 'purple': '紫'}
RGBA = {'red': [0.90, .06, .06, 1], 'green': [.04, .75, .12, 1], 'blue': [.04, .18, .95, 1],
        'purple': [.55, .20, .85, 1]}
SLOTS = [[.35, -.20, .025], [.50, -.20, .025], [.65, -.20, .025]]
BINS = {'red': [.32, .32], 'green': [.50, .32], 'blue': [.68, .32]}
HOLD = [.50, -.36]
HOME = [.40, .02, .44]
DT = 1 / 240
FPS = 12
EST_HEIGHT = .5
EST_FOV = 22
EST_SIZE = 224


def decide_hold(prediction, margin, threshold):
    """仕分けを保留するかの判定。未知色、または候補間差が小さい場合。"""
    if prediction not in COLORS:
        return True, 'unknown_color'
    if margin < threshold:
        return True, 'low_margin'
    return False, None


def px_to_world(px, py, center, height=EST_HEIGHT, fov=EST_FOV, size=EST_SIZE):
    """真上カメラの画素座標を作業面の座標へ変換する。符号と精度は検証手順で確認する。"""
    scale = 2 * height * math.tan(math.radians(fov / 2)) / size
    c = (size - 1) / 2
    return [float(center[0] + (px - c) * scale), float(center[1] - (py - c) * scale)]


def box(half, pos, rgba, mass=0):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(mass, collision, visual, pos)


def font_path(user=None):
    paths = [user, '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
             'C:/Windows/Fonts/meiryo.ttc', '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc']
    for path in paths:
        if path and Path(path).is_file():
            return path
    raise RuntimeError('日本語フォントが見つかりません。--font でフォントを指定してください。')


class Vision:
    def __init__(self):
        torch.set_num_threads(2)
        self.model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
        self.revision = self.model.config._commit_hash
        # モデルカードに従い、モデルへの候補文章だけは英語を用いる。
        # 未知色purpleは「どの色でもない」を表す候補で、保留判断に使う。
        self.prompts = [f'a photo of a {c} cube' for c in CANDIDATES]

    @torch.inference_mode()
    def classify(self, image):
        inputs = self.processor(text=self.prompts, images=image, return_tensors='pt', padding=True)
        logits = self.model(**inputs).logits_per_image[0]
        scores = logits.softmax(dim=-1).tolist()
        order = np.argsort(scores)
        label = CANDIDATES[int(order[-1])]
        return {'prediction': label, 'scores': dict(zip(CANDIDATES, scores)),
                'margin': scores[int(order[-1])] - scores[int(order[-2])],
                'logits': logits.tolist()}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.max_attempts = args.max_attempts
        self.grip_force = args.grip_force
        self.fault_active = False
        self.grip_scale = 1.0
        self.retry_events = 0
        self.active_attempt = 0
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.step = 0
        self.frames = 0
        self.phase = '準備'
        self.current = 0
        self.results = []
        self.predictions = []
        self.crops = []
        self.finger = .04
        self.successes = 0
        self.sorted_ok = 0
        self.holds = 0
        self.holds_ok = 0
        self.estimate_events = 0
        self.slip_events = 0
        self.z_max = 0.
        self.fonts = {}
        self.text_cache = {}
        if not args.no_video:
            if not shutil.which('ffmpeg'):
                raise RuntimeError('ffmpegをインストールしてください。')
            fp = font_path(args.font)
            self.fonts = {s: ImageFont.truetype(fp, s) for s in [17, 20, 24, 30, 36]}
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=150, deterministicOverlappingPairs=1)
        box([1.25, 1.0, .035], [.35, 0, -.035], [.80, .83, .86, 1])
        self.robot = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
        initial = [0, -.4, 0, -2.4, 0, 2.0, .8]
        for j, q in enumerate(initial):
            p.resetJointState(self.robot, j, q)
        for j in [9, 10]:
            p.resetJointState(self.robot, j, .04)
            p.changeDynamics(self.robot, j, lateralFriction=args.finger_friction, spinningFriction=.01)
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.target = np.array(HOME, dtype=float)
        # 初期化時のみ関節配置を設定。稼働中に物体や関節を瞬間移動させない。
        q = p.calculateInverseKinematics(self.robot, 11, HOME, self.orientation, maxNumIterations=300)
        for j in range(7):
            p.resetJointState(self.robot, j, q[j])
        for c, xy in BINS.items():
            x, y = xy
            rgba = RGBA[c]
            box([.076, .076, .006], [x, y, .006], [v * .5 + .5 for v in rgba[:3]] + [1])
            for dx, dy, hx, hy in [(-.08, 0, .004, .084), (.08, 0, .004, .084),
                                    (0, -.08, .076, .004), (0, .08, .076, .004)]:
                box([hx, hy, .023], [x + dx, y + dy, .023], rgba)
        # 初期色配置は実験条件。認識器にはこの配列を渡さない。
        self.colors = list(args.colors)
        self.truth = list(reversed(self.colors))
        if args.seed:
            self.truth = list(np.random.default_rng(args.seed).permutation(self.colors))
        # 保留皿。判定を保留した物体は色の皿へ運ばず、ここへ置く。
        box([.076, .076, .006], [HOLD[0], HOLD[1], .006], [.45, .47, .50, 1])
        if args.wall:
            # 障害物越え用の低い壁。経由点を挟む搬送の試験条件。
            box([.10, .006, .05], [.48, .06, .05], [.55, .58, .62, 1])
        for dx, dy, wxx, wyy in [(-.08, 0, .004, .084), (.08, 0, .004, .084),
                                 (0, -.08, .076, .004), (0, .08, .076, .004)]:
            box([wxx, wyy, .023], [HOLD[0] + dx, HOLD[1] + dy, .023], [.45, .47, .50, 1])
        self.objects = []
        for pos, color in zip(SLOTS, self.truth):
            body = box([.022, .022, .022], pos, RGBA[color], .06)
            p.changeDynamics(body, -1, lateralFriction=args.object_friction, spinningFriction=.01, restitution=0)
            self.objects.append(body)
        self.view = p.computeViewMatrixFromYawPitchRoll([.39, .03, .10], 1.4, 130, -48, 0, 2)
        self.proj = p.computeProjectionMatrixFOV(48, 1.6, .02, 4)
        self.advance(.6, record=False)

    def controls(self):
        q = p.calculateInverseKinematics(self.robot, 11, self.target.tolist(), self.orientation,
                                        maxNumIterations=100, residualThreshold=1e-5)
        p.setJointMotorControlArray(self.robot, list(range(7)), p.POSITION_CONTROL,
                                   targetPositions=q[:7], forces=[100] * 7,
                                   positionGains=[.12] * 7, velocityGains=[1] * 7)
        for j in [9, 10]:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=(.04 if self.fault_active else self.finger),
                                    force=self.grip_force * self.grip_scale, positionGain=.1)

    def advance(self, seconds, record=True):
        for _ in range(round(seconds / DT)):
            self.controls()
            p.stepSimulation()
            self.step += 1
            if record and self.writer and self.step % 20 == 0:
                self.capture()

    def move(self, pos, seconds, phase):
        self.phase = phase
        start = self.target.copy()
        end = np.array(pos, dtype=float)
        n = round(seconds / DT)
        for i in range(n):
            a = (i + 1) / n
            a = a * a * (3 - 2 * a)
            self.target = start + a * (end - start)
            self.advance(DT)

    def camera_crop(self, pos):
        x, y, _ = pos
        view = p.computeViewMatrix([x, y, .5], [x, y, 0], [0, 1, 0])
        proj = p.computeProjectionMatrixFOV(22, 1, .01, 1)
        rgba = p.getCameraImage(224, 224, view, proj, renderer=p.ER_TINY_RENDERER,
                               lightDirection=[1, -1, 3],
                               lightColor=[self.args.light_level] * 3, shadow=0)[2]
        return Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3])

    def estimate_position(self, center, body):
        """真上カメラから色付き塊の重心で物体位置を推定する。
        シミュレータ位置の観測は精度の評価だけに使い、推定には使わない。"""
        img = np.asarray(self.camera_crop([center[0], center[1], 0]), dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = ((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 40)).astype(np.uint8)
        count, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0 or int(areas.max()) < 30:
            return None
        best = 1 + int(np.argmax(areas))
        world = px_to_world(cents[best][0], cents[best][1], center)
        if abs(world[0] - center[0]) > .09 or abs(world[1] - center[1]) > .09:
            return None
        sim = list(p.getBasePositionAndOrientation(body)[0])
        return {'position_m': world, 'blob_pixels': int(stats[best, cv2.CC_STAT_AREA]),
                'sim_position_for_evaluation_m': sim}

    def mini_view(self, eye, look, size, fov, label):
        """ミニ窓用の小画像を取得する。推論には使わない録画専用の視点。"""
        view = p.computeViewMatrix(eye, look, [0, 1, 0] if eye[2] > .3 else [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov, size[0] / size[1], .01, 4)
        rgba = p.getCameraImage(size[0], size[1], view, proj, renderer=p.ER_TINY_RENDERER,
                                lightDirection=[-1, -2, 4], shadow=0)[2]
        img = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(size[1], size[0], 4)[:, :, :3])
        frame = Image.new('RGB', (size[0] + 2, size[1] + 2), '#79ddc7')
        frame.paste(img, (1, 1))
        d = ImageDraw.Draw(frame)
        fp = ImageFont.truetype(font_path(self.args.font), 13)
        d.text((4, 2), label, font=fp, fill='#0f2540')
        return frame

    def capture(self):
        rgba = p.getCameraImage(800, 500, self.view, self.proj, renderer=p.ER_TINY_RENDERER,
                               lightDirection=[-1, -2, 4], shadow=1)[2]
        scene = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(500, 800, 4)[:, :, :3])
        image = Image.new('RGB', (1280, 720), '#101d2d')
        image.paste(scene.resize((960, 600)), (0, 90))
        if self.args.multi_view:
            # 左下に真上（作業面全体）、その右に手先追従。録画専用で推論には使わない。
            tip = list(p.getLinkState(self.robot, 11)[0])
            overhead = self.mini_view([.47, -.05, .92], [.47, -.05, 0], (216, 216), 50, '真上')
            image.paste(overhead, (672, 90))
            eye = [tip[0], tip[1] - .35, tip[2] + .22]
            wrist = self.mini_view(eye, [tip[0], tip[1], max(0, tip[2] - .05)], (288, 176), 42, '手先')
            image.paste(wrist, (672, 316))
            d = ImageDraw.Draw(image)
            d.text((672, 692), '多視点: 俯瞰＋真上＋手先', font=self.fonts[17], fill='#9bc1d9')
        d = ImageDraw.Draw(image)
        def text(x, y, value, size=20, fill='#eef5fb'):
            # 日本語フォントの字形描画を繰り返さず、同じ字幕を再利用する。
            key = (value, size, fill)
            if key not in self.text_cache:
                bounds = self.fonts[size].getbbox(value)
                tile = Image.new('RGBA', (max(1, bounds[2] + 2), size * 2), (0, 0, 0, 0))
                ImageDraw.Draw(tile).text((0, 0), value, font=self.fonts[size], fill=fill)
                if len(self.text_cache) > 600:
                    self.text_cache.clear()
                self.text_cache[key] = tile
            tile = self.text_cache[key]
            image.paste(tile, (x, y), tile)
        text(24, 10, '画像と言葉で判断する、色仕分けロボット', 30)
        text(24, 52, 'PyBullet  ×  Panda  ×  CLIP   ｜   シミュレーション録画', 17, '#9bc1d9')
        d.rectangle((960, 90, 1279, 689), fill='#182b40')
        text(980, 108, '認識 → 判断 → 動作', 24)
        text(980, 152, f'対象 {self.current + 1}/3  試行 {self.active_attempt}/{self.max_attempts}', 20)
        text(980, 190, self.phase, 24, '#79ddc7')
        if self.predictions:
            info = self.predictions[min(self.current, len(self.predictions) - 1)]
            crop = self.crops[min(self.current, len(self.crops) - 1)]
            image.paste(crop.resize((112, 112)), (980, 242))
            text(1110, 246, '入力画像', 17)
            text(1110, 280, '判定：' + JP[info['prediction']], 24)
            text(1110, 316, '保留中' if info['held'] else '', 17, '#f0c862')
            text(980, 372, '候補文との相対スコア', 20)
            for k, color in enumerate(info['scores']):
                y = 414 + k * 30
                score = info['scores'][color]
                text(980, y, JP[color], 17)
                d.rectangle((1010, y + 5, 1010 + int(155 * score), y + 19),
                            fill=tuple(int(v * 255) for v in RGBA[color][:3]))
                text(1172, y, f'{score:.2f}', 17)
        text(980, 548, f'仕分け {self.sorted_ok}/3 ｜ 保留 {self.holds}/3', 20)
        text(980, 584, '色：画像・言語モデルで判定', 17)
        text(980, 612, '位置：治具＋再試行時は画像推定' if self.args.estimate
             else '位置：既知の供給治具を使用', 17)
        text(980, 640, '把持：接触と摩擦で搬送', 17)
        text(18, 695, f'等速・12コマ/秒 ｜ 物理時間 {self.step * DT:.1f}秒 ｜ 推論待ち時間は録画対象外', 17)
        self.writer.stdin.write(image.tobytes())
        self.frames += 1

    def observe(self, body):
        """シミュレータの観測。実機のセンサーや画像推定の代替ではない。"""
        pos = [float(v) for v in p.getBasePositionAndOrientation(body)[0]]
        contacts = {str(j): len(p.getContactPoints(self.robot, body, linkIndexA=j))
                    for j in [9, 10]}
        return {'position_m': pos, 'contacts': contacts,
                'bilateral_contact': all(v > 0 for v in contacts.values())}

    def lift_with_monitor(self, pos, seconds, phase, body):
        """移動しながら保持を監視する。すべり（走行最大高さからの低下）を検知したら
        把持力を1.6倍ずつ強める（上限2.5倍）。高さ0.2m未満への移動では判定しない。"""
        steps = max(1, round(seconds / .25))
        events, samples = [], []
        for k in range(steps):
            self.move(pos, seconds / steps, phase if k == 0 else self.phase)
            obs = self.observe(body)
            samples.append(obs)
            z = obs['position_m'][2]
            self.z_max = max(self.z_max, z)
            if not obs['bilateral_contact']:
                return {'held': False, 'samples': samples, 'events': events,
                        'observation': obs}
            if pos[2] > .2 and self.z_max - z > .008:
                old = self.grip_scale
                self.grip_scale = min(2.5, self.grip_scale * 1.6)
                events.append({'slip_m': round(self.z_max - z, 4),
                               'force_scale': [round(old, 2), round(self.grip_scale, 2)]})
                self.phase = 'すべり検知：把持力を強化'
                self.slip_events += 1
                self.z_max = z
                self.advance(.4)
        obs = self.observe(body)
        return {'held': obs['bilateral_contact'], 'samples': samples, 'events': events,
                'observation': obs}

    def grasp_attempt(self, xy, body, attempt, from_estimate=False):
        """把持を試みる。再試行の接近先は治具または推定位置。
        滑りやすい物体は弱く挟まないと指の間に押し出されるため、
        再試行では把持力を弱め、閉じる時間を伸ばす。"""
        x, y = xy
        self.active_attempt = attempt
        self.grip_scale = 1.0 if attempt == 1 else .4
        close_time = 2 if attempt == 1 else 3.5
        self.fault_active = (self.args.fault == 'always-open' or
                             (self.args.fault == 'first-open' and attempt == 1))
        injected = self.fault_active
        self.finger = .04
        if attempt == 1:
            approach = '供給位置へ接近'
        elif from_estimate:
            approach = '推定位置へ再接近'
        else:
            approach = '治具へ再接近'
        self.move([x, y, .26], 3, approach)
        self.move([x, y, .025], 2, '指を開いて下降')
        self.phase = '故障注入：指を開く' if injected else (
            '弱い力で静かに把持' if attempt > 1 else '接触・摩擦で把持')
        self.finger = 0
        self.advance(close_time)
        closed = self.observe(body)
        self.z_max = 0.
        lift = self.lift_with_monitor([x, y, .28], 3, '持ち上げ・すべり監視', body)
        samples = lift['samples']
        held = lift['held'] and lift['observation']['position_m'][2] > .12
        reason = None if held else ('no_bilateral_contact'
                                    if not lift['observation']['bilateral_contact']
                                    else 'insufficient_lift')
        self.fault_active = False
        self.grip_scale = 1.0
        return {'attempt': attempt, 'fault_injected': injected,
                'grasp_force_scale': .4 if attempt > 1 else 1.0,
                'closed_observation': closed, 'lift_samples': samples,
                'slip_events': lift['events'],
                'lift_height_m': lift['observation']['position_m'][2],
                'held': held, 'reason': reason}

    def recover(self, slot, body):
        """開いて退避。物体をリセットせず、治具内に残る場合のみ再挑戦可。"""
        self.fault_active = False
        self.finger = .04
        self.phase = '把持失敗：開放して待機'
        self.advance(1)
        self.move([slot[0], slot[1], .32], 1, '安全高さへ退避')
        obs = self.observe(body)
        x, y, z = obs['position_m']
        # 位置観測は再試行可否だけに使用。接近目標は変更しない。
        safe = (abs(x - slot[0]) < .012 and abs(y - slot[1]) < .012
                and .015 < z < .06)
        return {'observation': obs, 'safe_to_retry': safe,
                'reason': None if safe else 'outside_fixed_fixture'}

    def run(self, vision=None):
        started = time.time()
        vision = vision if vision is not None else Vision()
        for i, slot in enumerate(SLOTS):
            crop = self.camera_crop(slot)
            crop.save(self.out / f'input_{i + 1}.png')
            raw = vision.classify(crop)
            held, trigger = decide_hold(raw['prediction'], raw['margin'], self.args.hold_margin)
            info = {**raw, 'held': held, 'hold_trigger': trigger}
            self.crops.append(crop)
            self.predictions.append(info)
            print('認識', i + 1, info, flush=True)
        revision = vision.revision
        del vision
        if not self.args.no_video:
            self.writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(FPS),
                '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.out / 'demo.mp4')], stdin=subprocess.PIPE)
        try:
            self.phase = '画像から色を推論'
            self.advance(6)
            for i, (slot, body, info) in enumerate(zip(SLOTS, self.objects, self.predictions)):
                self.current = i
                destination = HOLD if info['held'] else BINS[info['prediction']]
                attempts = []
                held = False
                failure_reason = None
                current_target = list(slot[:2])
                estimate_used = False
                for attempt in range(1, self.max_attempts + 1):
                    if attempt > 1:
                        self.retry_events += 1
                    trial = self.grasp_attempt(current_target, body, attempt, estimate_used)
                    attempts.append(trial)
                    print('把持試行', i + 1, attempt, trial['held'], trial['reason'], flush=True)
                    if trial['held']:
                        held = True
                        break
                    if attempt == self.max_attempts:
                        failure_reason = trial['reason']
                        break
                    recovery = self.recover(current_target, body)
                    trial['recovery'] = recovery
                    failure_reason = trial['reason']
                    if recovery['safe_to_retry']:
                        current_target = list(slot[:2])
                        estimate_used = False
                        continue
                    if not self.args.estimate:
                        failure_reason = 'outside_fixed_fixture'
                        break
                    # 真上から撮る前に退避する。アームが視野内だと撮影を遮る。
                    self.move(HOME, 2, '視野を確保して退避')
                    est = self.estimate_position(current_target, body)
                    if est is None:
                        failure_reason = 'estimate_unavailable'
                        break
                    current_target = list(est['position_m'])
                    estimate_used = True
                    self.estimate_events += 1
                    trial['estimate'] = est
                if not held:
                    self.results.append({'slot': i + 1, 'prediction': info['prediction'],
                        'ground_truth_for_evaluation': self.truth[i], 'attempts': attempts,
                        'success': False, 'failure_reason': failure_reason,
                        'slip_events': sum(len(a['slip_events']) for a in attempts),
                        'policy_outcome': 'held' if info['held'] else 'sorted',
                        'estimate_used': estimate_used,
                        'final_position_m': self.observe(body)['position_m'],
                        'classification_correct': info['prediction'] == self.truth[i],
                        'inside_destination': False, 'recovered_after_retry': False})
                    self.phase = '上限到達または安全中止'
                    continue
                contacts = attempts[-1]['closed_observation']['contacts']
                lift = attempts[-1]['lift_height_m']
                dx, dy = destination
                if self.args.wall:
                    # 障害物越え: 壁上空の経由点を挟む２段階搬送。すべり監視は継続。
                    clearance = self.lift_with_monitor([.48, .06, .34], 3,
                                                       '壁上空の経由点へ', body)
                    if not (clearance['held'] and
                            clearance['observation']['position_m'][2] > .12):
                        self.finger = .04
                        self.phase = '搬送失敗：再把持せず中止'
                        self.advance(1)
                        self.move(HOME, 2, '退避')
                        self.results.append({'slot': i + 1,
                            'prediction': info['prediction'],
                            'ground_truth_for_evaluation': self.truth[i],
                            'attempts': attempts, 'success': False,
                            'failure_reason': 'transfer_lost',
                            'slip_events': sum(len(a['slip_events']) for a in attempts),
                            'clearance_slip_events': clearance['events'],
                            'policy_outcome': 'held' if info['held'] else 'sorted',
                            'estimate_used': estimate_used,
                            'transfer_observation': clearance['observation'],
                            'classification_correct': info['prediction'] == self.truth[i],
                            'inside_destination': False,
                            'final_position_m': self.observe(body)['position_m'],
                            'recovered_after_retry': False})
                        continue
                    transfer = self.lift_with_monitor([dx, dy, .28], 3,
                                                      '保留皿へ' if info['held'] else '認識した色の皿へ',
                                                      body)
                else:
                    transfer = self.lift_with_monitor([dx, dy, .28], 3,
                                                      '保留皿へ' if info['held'] else '認識した色の皿へ',
                                                      body)
                if not (transfer['held'] and transfer['observation']['position_m'][2] > .12):
                    self.finger = .04
                    self.phase = '搬送失敗：再把持せず中止'
                    self.advance(1)
                    self.move(HOME, 2, '退避')
                    self.results.append({'slot': i + 1, 'prediction': info['prediction'],
                        'ground_truth_for_evaluation': self.truth[i], 'attempts': attempts,
                        'success': False, 'failure_reason': 'transfer_lost',
                        'slip_events': sum(len(a['slip_events']) for a in attempts),
                        'transfer_slip_events': transfer['events'],
                        'policy_outcome': 'held' if info['held'] else 'sorted',
                        'estimate_used': estimate_used,
                        'transfer_observation': transfer['observation'],
                        'classification_correct': info['prediction'] == self.truth[i],
                        'inside_destination': False,
                        'final_position_m': self.observe(body)['position_m'],
                        'recovered_after_retry': False})
                    continue
                self.move([dx, dy, .065], 2, '保留皿へ下降' if info['held'] else '回収皿へ下降')
                self.phase = '指を開いて解放'
                self.finger = .04
                self.advance(2)
                self.move([dx, dy, .30], 2, '離れて結果を確認')
                final = p.getBasePositionAndOrientation(body)[0]
                inside = abs(final[0] - dx) < .065 and abs(final[1] - dy) < .065 and .02 < final[2] < .10
                correct = info['prediction'] == self.truth[i]
                # 保留が正解になるのは「読み色が正しく、安全のため保留した」場合。
                success = inside and correct and lift > .12 and all(contacts.values())
                self.successes += int(success)
                if info['held']:
                    self.holds += 1
                    self.holds_ok += int(success)
                else:
                    self.sorted_ok += int(success)
                if success:
                    outcome_reason = None
                elif not correct:
                    outcome_reason = 'held_incorrectly' if info['held'] else 'classification_error'
                else:
                    outcome_reason = 'placement_error'
                result = {'slot': i + 1, 'prediction': info['prediction'],
                          'ground_truth_for_evaluation': self.truth[i],
                          'contacts_before_lift': contacts, 'lift_height_m': lift,
                          'final_position_m': final, 'inside_destination': inside,
                          'policy_outcome': 'held' if info['held'] else 'sorted',
                          'hold_trigger': info['hold_trigger'], 'estimate_used': estimate_used,
                          'classification_correct': correct, 'success': success,
                          'attempts': attempts,
                          'slip_events': sum(len(a['slip_events']) for a in attempts),
                          'transfer_slip_events': transfer['events'],
                          'transfer_observation': transfer['observation'],
                          'failure_reason': outcome_reason,
                          'recovered_after_retry': success and len(attempts) > 1}
                self.results.append(result)
                print('結果', json.dumps(result, ensure_ascii=False), flush=True)
            self.move(HOME, 3, '初期位置へ復帰')
            self.phase = '終了：結果を確認'
            self.advance(8)
        finally:
            if self.writer:
                self.writer.stdin.close()
                code = self.writer.wait()
                if code:
                    raise RuntimeError(f'ffmpegが終了コード{code}で失敗しました')
            versions = {name: importlib.metadata.version(name) for name in
                        ['pybullet', 'torch', 'transformers', 'numpy', 'pillow']}
            report = {'model': MODEL_ID, 'model_revision': revision, 'versions': versions,
                      'seed': self.args.seed, 'predictions': self.predictions, 'results': self.results,
                      'configuration': vars(self.args), 'retry_events': self.retry_events,
                      'recovered_objects': sum(r['recovered_after_retry'] for r in self.results),
                      'first_attempt_grasps': sum(bool(r['attempts'][0]['held']) for r in self.results),
                      'estimate_events': self.estimate_events,
                      'slip_events': self.slip_events,
                      'held_objects': self.holds, 'hold_correct': self.holds_ok,
                      'sorted_correct': self.sorted_ok,
                      'successful_sorts': self.successes, 'total_objects': 3,
                      'video_frames': self.frames, 'fps': FPS,
                      'video_duration_seconds': self.frames / FPS,
                      'simulation_seconds': self.step * DT, 'wall_seconds': time.time() - started,
                      'assumptions': ['初回の供給位置は既知治具', '分類はCLIP推論', '物体移動は接触・摩擦のみ',
                                      '相対スコアは校正済みの正解確率ではない', '推論待ちは録画対象外',
                                      '位置推定は真上カメラの古典的な画像処理', '保留判断は未知色と低い候補間差が条件',
                                      'すべり検知は高さ低下の監視と把持力の段階的強化']}
            (self.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            p.disconnect()
        if self.successes != 3 and not self.args.allow_failures:
            raise RuntimeError(f'仕分け成功は{self.successes}/3件。result.jsonを確認してください。')
        return report


def build_args(**overrides):
    """試験・二次利用向けの既定設定。録画なし・失敗許容が初期値。"""
    base = dict(output='output', no_video=True, font=None, seed=0, max_attempts=2,
                fault='none', colors=['red', 'green', 'blue'], hold_margin=.5,
                estimate=True, finger_friction=2., object_friction=1.5, grip_force=35.,
                light_level=1., allow_failures=True, multi_view=False, wall=False)
    base.update(overrides)
    return Namespace(**base)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='output')
    parser.add_argument('--no-video', action='store_true', help='録画なしで物理動作を試験')
    parser.add_argument('--font', help='日本語を表示できるフォントのパス')
    parser.add_argument('--seed', type=int, default=0, help='0は固定配置。その他は色配置の入れ替え')
    parser.add_argument('--max-attempts', type=int, choices=range(1, 5), default=2,
                        help='初回を含む最大把持回数（１〜４）')
    parser.add_argument('--fault', choices=['none', 'first-open', 'always-open'], default='none',
                        help='試験用故障注入。first-openは各物体の初回だけ指を開いたままにする')
    parser.add_argument('--colors', default='red,green,blue',
                        help='供給する３物体の色。purpleを含めると保留判断の対象になる')
    parser.add_argument('--hold-margin', type=float, default=.5,
                        help='候補間差がこの値未満なら仕分けを保留する（0〜1）')
    parser.add_argument('--no-estimate', action='store_true',
                        help='再試行時の画像位置推定を無効化する（旧動作）')
    parser.add_argument('--multi-view', action='store_true',
                        help='俯瞰＋真上＋手先の３視点を合成して録画する')
    parser.add_argument('--wall', action='store_true',
                        help='障害物越え: 供給列と回収皿の間に低い壁を置く')
    parser.add_argument('--finger-friction', type=float, default=2.0)
    parser.add_argument('--object-friction', type=float, default=1.5)
    parser.add_argument('--grip-force', type=float, default=35.0)
    parser.add_argument('--light-level', type=float, default=1.0,
                        help='認識用カメラだけの光の色成分（0.1〜1）')
    parser.add_argument('--allow-failures', action='store_true',
                        help='物理的失敗は記録して終了コード０。実装例外は隠さない')
    args = parser.parse_args()
    args.colors = [c.strip().lower() for c in args.colors.split(',')]
    if len(args.colors) != 3 or any(c not in CANDIDATES for c in args.colors):
        parser.error('colorsはred,green,blue,purpleから３色です')
    args.estimate = not args.no_estimate
    if not all(math.isfinite(v) and v >= 0 for v in
               [args.finger_friction, args.object_friction, args.grip_force]):
        parser.error('摩擦と力は有限の非負値にしてください')
    if not math.isfinite(args.hold_margin) or not 0 <= args.hold_margin <= 1:
        parser.error('hold-marginは0〜1にしてください')
    if not math.isfinite(args.light_level) or not .1 <= args.light_level <= 1:
        parser.error('light-levelは0.1〜1にしてください')
    Experiment(args).run()


if __name__ == '__main__':
    main()

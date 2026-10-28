#!/usr/bin/env python3
"""校正用プローブ。CLIPの候補間差の測定と、位置推定の精度確認。

* 離散的な配置替えは試験条件の設定であり、ロボット動作中の偽装ではない。
* 結果は calibration_probe.json に保存し、しきい値と符号の決定根拠にする。
"""
import json
from pathlib import Path
import numpy as np
import pybullet as p
from run_sorting import Experiment, SLOTS, Vision, build_args


def margin_probe(vision):
    """標準色と未知色の候補間差を、通常光と薄暗い光で測る。"""
    results = {}
    for label, colors, light in [('normal', ['red', 'green', 'blue'], 1.),
                                 ('normal_dim', ['red', 'green', 'blue'], .15),
                                 ('purple', ['red', 'purple', 'green'], 1.),
                                 ('purple_dim', ['red', 'purple', 'green'], .15)]:
        exp = Experiment(build_args(output='probe_margins', light_level=light, colors=colors))
        try:
            # purple条件では中央（purple）の判定を、それ以外は最初の物体を見る。
            index = 1 if 'purple' in colors else 0
            info = vision.classify(exp.camera_crop(SLOTS[index]))
            results[label] = {'prediction': info['prediction'], 'margin': info['margin'],
                              'scores': info['scores']}
        finally:
            if p.isConnected():
                p.disconnect()
    return results


def estimation_probe():
    """既知のずれ量に対する推定誤差を測る。配置替えは試験条件の設定。"""
    exp = Experiment(build_args(output='probe_estimation'))
    body = exp.objects[0]
    slot = SLOTS[0]
    trials = []
    for dx, dy in [(0, 0), (.03, 0), (-.03, .02), (0, -.04), (.06, .03)]:
        # 試験条件の設定として物体を配置替えする。ロボット動作中の瞬間移動ではない。
        p.resetBasePositionAndOrientation(body, [slot[0] + dx, slot[1] + dy, .034],
                                          [0, 0, 0, 1])
        exp.advance(.3, record=False)
        est = exp.estimate_position(slot, body)
        truth = [slot[0] + dx, slot[1] + dy]
        if est is None:
            trials.append({'offset_m': [dx, dy], 'truth_m': truth, 'estimated': False})
            continue
        error = float(np.hypot(est['position_m'][0] - truth[0],
                               est['position_m'][1] - truth[1]))
        trials.append({'offset_m': [dx, dy], 'truth_m': truth, 'estimated': True,
                       'estimate_m': est['position_m'], 'error_m': error,
                       'blob_pixels': est['blob_pixels']})
    if p.isConnected():
        p.disconnect()
    ok = all(t['estimated'] and t['error_m'] < .01 for t in trials)
    return {'trials': trials, 'estimation_ok': ok,
            'max_error_m': max((t.get('error_m', 1) for t in trials), default=1)}


def main():
    decision = margin_probe(Vision())
    estimation = estimation_probe()
    report = {'decision_margins': decision, 'estimation': estimation,
              'note': '小規模な校正用測定。しきい値と符号の確認に使う。'}
    Path('calibration_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not estimation['estimation_ok']:
        raise SystemExit('位置推定の精度または符号を確認してください')


if __name__ == '__main__':
    main()

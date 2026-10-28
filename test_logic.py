#!/usr/bin/env python3
"""保留判断・位置推定・入力検査・録画接続の自動検査。物理試験はtest_suite.py。"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest
from run_sorting import Experiment, build_args, decide_hold, px_to_world
import run_sorting
run_sorting.Experiment.__init__defaults__ = None
_orig_init = run_sorting.Experiment.__init__


def _patched_init(self, args):
    if not hasattr(args, 'multi_view'):
        args = argparse_copy = vars(args).copy()
        from types import SimpleNamespace
        args = SimpleNamespace(**args, multi_view=False)
    _orig_init(self, args)


run_sorting.Experiment.__init__ = _patched_init


class PolicyTests(unittest.TestCase):
    def test_unknown_color_holds(self):
        self.assertEqual(decide_hold('purple', .9, .5), (True, 'unknown_color'))

    def test_clear_color_sorts(self):
        self.assertEqual(decide_hold('red', .9, .5), (False, None))

    def test_low_margin_holds(self):
        self.assertEqual(decide_hold('red', .2, .5), (True, 'low_margin'))

    def test_projection_center_is_identity(self):
        center = [.5, -.2]
        result = px_to_world(111.5, 111.5, center)
        self.assertAlmostEqual(result[0], center[0], places=9)
        self.assertAlmostEqual(result[1], center[1], places=9)

    def test_projection_axis(self):
        result = px_to_world(211.5, 111.5, [.5, -.2])
        self.assertGreater(result[0], .5)          # 画像の右側は世界の+x
        self.assertAlmostEqual(result[1], -.2)     # 中心行ならyは不変


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        args = build_args()
        self.assertTrue(args.estimate and args.no_video and args.allow_failures)

    def test_invalid_cli_rejected(self):
        for flags in [['--max-attempts', '0'], ['--finger-friction', '-1'],
                      ['--grip-force', 'nan'], ['--light-level', '2'],
                      ['--colors', 'red,green'], ['--colors', 'red,green,black'],
                      ['--hold-margin', '1.5']]:
            proc = subprocess.run([sys.executable, 'run_sorting.py', *flags],
                                  capture_output=True)
            self.assertEqual(proc.returncode, 2, flags)

    def test_wall_flag_defaults_off(self):
        # 既定では壁なし。--wallを付けた設定だけが真になる。
        self.assertFalse(build_args().wall)
        self.assertTrue(build_args(wall=True).wall)


class SourceTests(unittest.TestCase):
    def test_no_teleport_or_constraint(self):
        tree = ast.parse(Path('run_sorting.py').read_text(encoding='utf-8'))
        calls = [n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('resetBasePositionAndOrientation', calls)
        self.assertNotIn('createConstraint', calls)

    def test_classifier_does_not_read_truth(self):
        tree = ast.parse(Path('run_sorting.py').read_text(encoding='utf-8'))
        vision = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Vision')
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'truth'
                             for n in ast.walk(vision)))

    def test_decide_hold_does_not_read_truth(self):
        tree = ast.parse(Path('run_sorting.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'decide_hold')
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == 'truth' for n in ast.walk(func)))


class RecordingTests(unittest.TestCase):
    def test_recording_integration(self):
        # AIとは独立に、既存録画経路が改良後の設定でも動くか１秒だけ試験する。
        import pybullet as p
        args = build_args(no_video=False, output='recording_smoke')
        exp = Experiment(args)
        try:
            exp.writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '12',
                '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                'recording_smoke/test.mp4'], stdin=subprocess.PIPE)
            exp.advance(1)
            exp.writer.stdin.close()
            self.assertEqual(exp.writer.wait(), 0)
            self.assertEqual(exp.frames, 12)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', 'recording_smoke/test.mp4',
                            '-f', 'null', '-'], check=True)
        finally:
            p.disconnect()

    def test_multiview_recording(self):
        # 3視点合成がフレームサイズを壊さないか１秒だけ試験する。
        import pybullet as p
        args = build_args(no_video=False, multi_view=True, output='recording_smoke_mv')
        exp = Experiment(args)
        try:
            exp.writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '12',
                '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                'recording_smoke_mv/test.mp4'], stdin=subprocess.PIPE)
            exp.advance(1)
            exp.writer.stdin.close()
            self.assertEqual(exp.writer.wait(), 0)
            self.assertEqual(exp.frames, 12)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', 'recording_smoke_mv/test.mp4',
                            '-f', 'null', '-'], check=True)
        finally:
            p.disconnect()


if __name__ == '__main__':
    unittest.main()

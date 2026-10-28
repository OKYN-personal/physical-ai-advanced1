#!/usr/bin/env python3
"""録画ファイルの全フレームを復号し、動作ログの成功条件を再計算する。"""
import argparse
import json
import math
from pathlib import Path
import subprocess

BINS = {'red': [.32, .32], 'green': [.50, .32], 'blue': [.68, .32]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output')
    args = parser.parse_args()
    out = Path(args.output)
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_read_frames,avg_frame_rate,codec_name:format=duration',
        '-of', 'json', str(out / 'demo.mp4')], text=True))
    stream = probe['streams'][0]
    duration = float(probe['format']['duration'])
    assert 60 <= duration <= 180, duration
    assert (stream['width'], stream['height']) == (1280, 720)
    assert int(stream['nb_read_frames']) == result['video_frames']
    assert abs(duration - result['video_duration_seconds']) < .1
    assert stream['codec_name'] == 'h264'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(out / 'demo.mp4'),
                    '-f', 'null', '-'], check=True)
    passed = []
    for r in result['results']:
        bx, by = BINS[r['prediction']]
        x, y, z = r['final_position_m']
        verdict = (abs(x - bx) < .065 and abs(y - by) < .065 and .02 < z < .10
                   and r['prediction'] == r['ground_truth_for_evaluation']
                   and r['lift_height_m'] > .12
                   and all(c > 0 for c in r['contacts_before_lift'].values()))
        assert verdict == r['success']
        passed.append(verdict)
    assert len(passed) == 3 and all(passed)
    assert result['successful_sorts'] == sum(passed)
    for info in result['predictions']:
        assert max(info['scores'], key=info['scores'].get) == info['prediction']
        assert math.isclose(sum(info['scores'].values()), 1.0, abs_tol=1e-5)
    print(json.dumps({'動画の長さ_秒': duration, '解像度': [stream['width'], stream['height']],
                      '復号フレーム数': int(stream['nb_read_frames']),
                      '成功条件の再計算': passed, '全フレーム復号': True}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

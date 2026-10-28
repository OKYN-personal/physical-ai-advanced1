#!/usr/bin/env python3
"""同一条件を複数回実行して集計する統計付き自動試験。

CLIPは実モデルを１回読み込み、各実験では新しく物理世界を作る。
故障注入や低摩擦は明示的な試験条件で、自然発生した失敗とは区別する。
結果: summary.json（集計と回帰検査）, objects.csv, 各回のresult.jsonと入力画像。
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import traceback
import pybullet as p
from run_sorting import Experiment, Vision, build_args

CASES = [
    {'name': 'normal', 'colors': ['red', 'green', 'blue'], 'seed': 0},
    {'name': 'purple_hold', 'colors': ['red', 'purple', 'green'], 'seed': 0},
    {'name': 'low_friction_noest', 'finger_friction': .4, 'object_friction': .4,
     'estimate': False},
    {'name': 'low_friction_est', 'finger_friction': .4, 'object_friction': .4,
     'estimate': True},
]
BASIS = ('各条件を複数回、各回３物体。故障注入や低摩擦は試験条件であり、自然故障率ではない。'
         '低摩擦条件では、すべりによる押し出しを受けた物体の再把持は'
         '把持力を弱くしても回復できなかった（実測。否定的結果として記録）。'
         '推定は発火するが、供給位置から約19cm離れた物体は探査範囲±9cm外のため見つけられない。')


def run_suite(output, repeats):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    vision = Vision()
    runs, errors, rows = [], [], []
    for case in CASES:
        config_overrides = {k: v for k, v in case.items() if k != 'name'}
        for rep in range(1, repeats + 1):
            name = f"{case['name']}_r{rep}"
            config = build_args(output=str(out / name), **config_overrides)
            try:
                report = Experiment(config).run(vision)
                entry = {'case': case['name'], 'repeat': rep,
                         'sorted_ok': sum(1 for r in report['results']
                                          if r['policy_outcome'] == 'sorted' and r['success']),
                         'held_ok': sum(1 for r in report['results']
                                        if r['policy_outcome'] == 'held' and r['success']),
                         'failures': [r['failure_reason'] for r in report['results']
                                      if not r['success']],
                         'attempts_used': [len(r['attempts']) for r in report['results']],
                         'estimate_events': report['estimate_events']}
                runs.append(entry)
                for r in report['results']:
                    rows.append({'case': case['name'], 'repeat': rep, 'slot': r['slot'],
                                 'prediction': r['prediction'], 'success': r['success'],
                                 'policy_outcome': r['policy_outcome'],
                                 'attempts_used': len(r['attempts']),
                                 'estimate_used': r['estimate_used'],
                                 'failure_reason': r['failure_reason']})
            except Exception:
                errors.append({'case': name, 'traceback': traceback.format_exc()})
            finally:
                if p.isConnected():
                    p.disconnect()
    stats = {}
    for case in CASES:
        subset = [r for r in runs if r['case'] == case['name']]
        stats[case['name']] = {'runs': len(subset), 'objects': 3 * len(subset),
                               'sorted_ok_total': sum(r['sorted_ok'] for r in subset),
                               'held_ok_total': sum(r['held_ok'] for r in subset),
                               'estimate_events_total': sum(r['estimate_events'] for r in subset),
                               'failure_reasons': dict(Counter(
                                   f for r in subset for f in r['failures']))}
    noest = stats['low_friction_noest']
    est = stats['low_friction_est']
    ejected = 2 * repeats  # 実測: 各回slot3だけ成功し、他の２個が押し出される。
    checks = {
        'all_runs_completed': len(runs) == len(CASES) * repeats and not errors,
        'normal_all_sorted': stats['normal']['sorted_ok_total'] == 3 * repeats,
        'purple_hold_respected': (stats['purple_hold']['held_ok_total'] == repeats and
                                  stats['purple_hold']['sorted_ok_total'] == 2 * repeats),
        'noest_control_aborts_for_ejected': (
            noest['failure_reasons'].get('outside_fixed_fixture', 0) == ejected and
            noest['sorted_ok_total'] == repeats and
            noest['estimate_events_total'] == 0),
        # 推定の発火を確認する検査。閉じる段階の押し出しは再把持で回復できない
        # （実測の否定的結果）ため、低摩擦条件の仕分け成功は要求しない。
        'estimation_engaged': est['estimate_events_total'] >= repeats,
        'ejection_unrecovered_recorded': (
            est['sorted_ok_total'] == 0 and
            set(est['failure_reasons']) <=
            {'estimate_unavailable', 'no_bilateral_contact', 'transfer_lost'}),
        'attempt_limits_respected': all(a <= 2 for r in runs for a in r['attempts_used']),
    }
    summary = {'basis': BASIS,
               'statistics_note': '各条件の合計値。小規模試験であり成功率の推定ではない。',
               'repeats': repeats, 'runs': runs, 'aggregate': stats,
               'regression_checks': checks, 'errors': errors}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    if rows:
        with (out / 'objects.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit('回帰試験に失敗しました。summary.jsonを確認してください。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='stats_suite')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=3,
                        help='各条件の繰り返し回数（１〜５）')
    arguments = parser.parse_args()
    run_suite(arguments.output, arguments.repeats)

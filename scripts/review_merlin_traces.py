#!/usr/bin/env python3
"""
Structured review of Merlin game traces.

Reads all Merlin JSONL files from the two relevant run directories, emits one
human-readable block per game, sorted: losses first (grouped by end reason),
then wins. Pipe to `less` or redirect to a file for review.

Usage:
    python scripts/review_merlin_traces.py [--runs-dir data/runs]
    python scripts/review_merlin_traces.py | less
    python scripts/review_merlin_traces.py > /tmp/merlin_review.txt
"""
import argparse
import json
from pathlib import Path

MERLIN_RUN_PREFIXES = [
    'avalon_e1_merlin__',
    'avalon_e1_merlin_procedural__',
    'avalon_e1_merlin_minimal_disc__',
]


def load_game(fpath: Path) -> dict | None:
    lines = fpath.read_text().splitlines()
    if not lines:
        return None
    events = []
    for line in lines:
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        return None

    by_type = {}
    for ev in events:
        by_type.setdefault(ev['event'], []).append(ev)

    trial_start = next((e for e in events if e['event'] == 'trial_start'), None)
    game_start  = next((e for e in events if e['event'] == 'game_start'), None)
    game_end    = next((e for e in events if e['event'] == 'game_end'), None)
    if not trial_start or not game_start or not game_end:
        return None

    cond        = trial_start['condition']
    role_names  = game_start['role_names']
    evil_idxs   = frozenset(i for i, r in enumerate(role_names) if r in ('Assassin', 'Minion'))
    good_idxs   = frozenset(i for i, r in enumerate(role_names) if r not in ('Assassin', 'Minion'))

    # Index vote results by (quest_turn, round)
    vote_by_round = {}
    for ev in events:
        if ev['event'] == 'team_vote_result':
            vote_by_round[(ev['quest_turn'], ev['round'])] = ev

    # Index quest outcomes
    quest_results = {ev['quest_turn']: ev for ev in events if ev['event'] == 'quest_result'}

    # All proposals, in order
    proposals = [ev for ev in events if ev['event'] == 'team_proposed']

    # Assassination event (only if good wins 3 quests)
    assassination = next((e for e in events if e['event'] == 'assassination'), None)

    good_wins    = game_end.get('good_wins', False)
    n_good_quests = game_end.get('n_good_quests', 0)
    n_quests_played = len(quest_results)

    if good_wins:
        if assassination and not assassination.get('good_wins', True):
            end_reason = 'assassinated'
        else:
            end_reason = 'good_win'
    elif n_quests_played > 0:
        end_reason = 'evil_quests'
    else:
        end_reason = 'unknown'

    return {
        'trial_id':       trial_start['trial_id'],
        'run_name':       fpath.parent.name.split('__')[0],
        'cond':           cond,
        'role_names':     role_names,
        'evil_idxs':      evil_idxs,
        'good_idxs':      good_idxs,
        'proposals':      proposals,
        'vote_by_round':  vote_by_round,
        'quest_results':  quest_results,
        'assassination':  assassination,
        'good_wins':      good_wins,
        'n_good_quests':  n_good_quests,
        'n_quests_played':n_quests_played,
        'end_reason':     end_reason,
        'game_end':       game_end,
    }


def format_team(team: list, evil_idxs: frozenset) -> str:
    parts = []
    for p in sorted(team):
        tag = '*EVIL*' if p in evil_idxs else 'good'
        parts.append(f"P{p}({tag})")
    return '[' + ', '.join(parts) + ']'


def render_game(g: dict, idx: int) -> str:
    lines = []
    cond = g['cond']
    lines.append(f"{'='*72}")
    lines.append(f"Game {idx:03d}  |  {g['trial_id']}  |  {g['run_name']}")
    lines.append(f"  Role: {cond['llm_role']}  Budget: {cond['budget']}  "
                 f"Prompt: {cond['prompt_variant']}  "
                 f"Disc: {cond.get('with_discussion', False)}  Seed: {cond['seed']}")
    lines.append(f"  Roles: {g['role_names']}  "
                 f"Evil players: {sorted(g['evil_idxs'])}")
    lines.append(f"  Result: {'GOOD WINS' if g['good_wins'] else 'EVIL WINS'}  "
                 f"End: {g['end_reason']}  "
                 f"Good quests: {g['n_good_quests']}/{g['n_quests_played']}")
    lines.append('')

    # Quest sequence
    lines.append(f"  Quest sequence:")
    for qt in sorted(g['quest_results']):
        qr   = g['quest_results'][qt]
        won  = qr['succeeded']
        team = qr['team']
        evil_on = [p for p in team if p in g['evil_idxs']]
        clean   = len(evil_on) == 0
        tag  = 'PASS' if won else 'FAIL'
        etag = '' if clean else f'  ← evil on team: {evil_on}'
        lines.append(f"    Q{qt}: [{tag}]  team={sorted(team)}{etag}")
    lines.append('')

    # LLM proposal history
    llm_proposals = [p for p in g['proposals'] if p.get('is_llm_decision')]
    lines.append(f"  LLM proposals ({len(llm_proposals)} total):")
    for ev in llm_proposals:
        qt    = ev['quest_turn']
        rnd   = ev['round']
        team  = ev['team']
        vote  = g['vote_by_round'].get((qt, rnd))
        acc   = vote['accepted'] if vote else None
        evil_on = [p for p in team if p in g['evil_idxs']]
        clean   = len(evil_on) == 0

        qr    = g['quest_results'].get(qt)
        quest_tag = ''
        if acc and qr:
            quest_tag = f"  → quest {'PASS' if qr['succeeded'] else 'FAIL'}"

        acc_tag = {True: 'ACCEPTED', False: 'REJECTED', None: '??'}[acc]
        p0_tag  = ' +P0' if 0 in team else ''
        dirty_tag = f'  ← EVIL INCLUDED: {evil_on}' if not clean else ''
        lines.append(
            f"    Q{qt}/R{rnd}: {format_team(team, g['evil_idxs'])}{p0_tag}"
            f"  [{acc_tag}]{quest_tag}{dirty_tag}"
        )

    lines.append('')

    # Assassination (if any)
    if g['assassination']:
        a = g['assassination']
        hit = a.get('target_is_merlin', None)
        hit_tag = 'HIT MERLIN' if hit else 'missed'
        lines.append(f"  Assassination: P{a['assassin_idx']} → P{a['target']}  [{hit_tag}]  "
                     f"Good wins: {a.get('good_wins', '?')}")
        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='data/runs',
                        help='Path to the runs directory (default: data/runs)')
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        # Try relative to script location
        runs_dir = Path(__file__).parent.parent / 'data' / 'runs'

    merlin_dirs = [
        d for d in sorted(runs_dir.iterdir())
        if d.is_dir() and any(d.name.startswith(p) for p in MERLIN_RUN_PREFIXES)
    ]

    if not merlin_dirs:
        print(f"No Merlin run directories found in {runs_dir}")
        return

    games = []
    for run_dir in merlin_dirs:
        for fpath in sorted(run_dir.glob('trial_*.jsonl')):
            g = load_game(fpath)
            if g:
                games.append(g)

    if not games:
        print("No valid game files found.")
        return

    # Summary header
    n_total   = len(games)
    n_wins    = sum(1 for g in games if g['good_wins'])
    n_assassinated = sum(1 for g in games if g['end_reason'] == 'assassinated')
    n_evil_quests  = sum(1 for g in games if g['end_reason'] == 'evil_quests')
    n_unknown      = sum(1 for g in games if g['end_reason'] == 'unknown')

    print(f"{'='*72}")
    print(f"MERLIN TRACE REVIEW — {n_total} games")
    print(f"  Wins:          {n_wins}")
    print(f"  Assassinated:  {n_assassinated}  (good reached 3 quests, then Merlin guessed)")
    print(f"  Evil quests:   {n_evil_quests}  (evil won 3 quests — structural failure mode)")
    print(f"  Unknown:       {n_unknown}")
    print(f"  Run dirs: {[d.name for d in merlin_dirs]}")
    print(f"{'='*72}")
    print()

    # Sort: losses first (evil_quests, then assassinated, then unknown), then wins
    order = {'evil_quests': 0, 'assassinated': 1, 'unknown': 2, 'good_win': 3}
    games_sorted = sorted(games, key=lambda g: (order.get(g['end_reason'], 9), g['trial_id']))

    for idx, g in enumerate(games_sorted, 1):
        print(render_game(g, idx))


if __name__ == '__main__':
    main()

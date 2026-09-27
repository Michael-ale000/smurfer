#!/usr/bin/env python
"""
Package one agent for the tournament, after checking the things that actually go wrong.

The graders unzip the archive, install anything in a `requirements.txt` they find,
take **the first directory containing a callbacks.py**, drop it into their own
untouched copy of the framework, and run one game with ``self.train = False`` against
three ``random_agent``s.  Everything this script checks follows from that:

  * the trained parameters must be inside the agent directory (nothing else travels);
  * nothing may reference a path outside the agent directory, or import the training
    driver, the test suite, or anything else that stays on your machine;
  * the agent must survive a game with ``--train 0``, which is the one configuration
    that is easy to forget to test while developing under ``--train 1``;
  * no multiprocessing in the agent (explicitly forbidden by the project sheet);
  * junk (``__pycache__``, logs, ``.bak`` checkpoints) should not be shipped.

Usage:
    python3 make_submission.py --agent dqn_agent
    python3 make_submission.py --agent my_agent --name teamname_tabular
"""

import argparse
import ast
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
EXCLUDE_DIRS = {'__pycache__', 'logs', '.ipynb_checkpoints'}
EXCLUDE_SUFFIXES = {'.bak', '.log', '.pyc'}
MODEL_SUFFIXES = {'.pt', '.pth', '.pkl', '.npy', '.npz', '.joblib', '.h5', '.json'}

# Modules the framework provides; anything else imported at the top level of the agent
# has to come from requirements.txt.
FRAMEWORK_MODULES = {'settings', 'events', 'items', 'environment', 'fallbacks', 'agents'}


class Report:
    def __init__(self):
        self.problems, self.warnings = [], []

    def fail(self, msg):
        self.problems.append(msg)
        print(f'  FAIL  {msg}')

    def warn(self, msg):
        self.warnings.append(msg)
        print(f'  WARN  {msg}')

    def ok(self, msg):
        print(f'  ok    {msg}')


def collect_files(agent_dir):
    for path in sorted(agent_dir.rglob('*')):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(agent_dir).parts):
            continue
        if path.suffix in EXCLUDE_SUFFIXES or path.name.startswith('.'):
            continue
        yield path


def check_agent(agent_dir, files, report):
    names = {p.name for p in files}

    if 'callbacks.py' not in names:
        report.fail('no callbacks.py -- the graders locate your agent by this file')
    else:
        report.ok('callbacks.py present')

    models = [p for p in files if p.suffix in MODEL_SUFFIXES]
    if models:
        total = sum(p.stat().st_size for p in models) / 1e6
        report.ok(f"trained parameters: {', '.join(p.name for p in models)} ({total:.1f} MB)")
    else:
        report.fail('no trained parameters in the directory -- an untrained agent '
                    'plays at random in the tournament')

    for path in files:
        if path.suffix != '.py':
            continue
        source = path.read_text()
        rel = path.relative_to(agent_dir)

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            report.fail(f'{rel} does not parse: {exc}')
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [] if node.level else [(node.module or '').split('.')[0]]
            else:
                continue
            for mod in mods:
                if mod == 'multiprocessing':
                    report.fail(f'{rel} imports multiprocessing, which the project '
                                f'sheet forbids in the submitted agent')
                if mod.startswith('agent_code'):
                    report.fail(f'{rel} imports {mod} -- an agent must not depend on '
                                f'anything outside its own directory')

        # An absolute path baked into the code is the classic submission failure: it
        # points at your machine and does not exist on theirs.
        for match in re.finditer(r"""['"](/[^'"\n]{3,})['"]""", source):
            report.fail(f'{rel} contains the absolute path {match.group(1)!r}')

    return names


def check_requirements(report):
    req = ROOT / 'requirements.txt'
    if req.exists():
        report.ok(f'requirements.txt found; it will be included in the archive')
        return req
    report.warn('no requirements.txt at the repository root -- include one if your '
                'agent needs a library beyond the tournament image')
    return None


def play_one_game(agent_name, report):
    """Exactly the run the graders make: one game, no training, three random_agents."""
    cmd = [sys.executable, 'main.py', 'play', '--no-gui',
           '--agents', agent_name, 'random_agent', 'random_agent', 'random_agent',
           '--n-rounds', '1']
    print(f"\n  running: {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        report.fail(f'the graders\' run crashed (exit {result.returncode}):\n'
                    f'{result.stderr.strip()[-1500:]}')
        return

    log = ROOT / 'agent_code' / agent_name / 'logs'
    errors = []
    for path in log.glob('*.log'):
        errors += [line for line in path.read_text().splitlines()
                   if 'ERROR' in line or 'WARNING' in line]
    if errors:
        report.warn('the agent log contains warnings/errors:')
        for line in errors[:5]:
            print(f'          {line}')
    else:
        report.ok('one untrained-mode game against three random_agents, no errors')


def build_zip(agent_dir, files, name, req, out):
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, Path(name) / path.relative_to(agent_dir))
        if req is not None:
            z.write(req, 'requirements.txt')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True, help='directory under agent_code/')
    parser.add_argument('--name', default=None,
                        help='folder name inside the zip; this is the name your agent '
                             'plays under in the tournament (default: --agent)')
    parser.add_argument('--out', default='final-project-agent-code.zip')
    parser.add_argument('--no-check', action='store_true',
                        help='skip the one-game test run')
    args = parser.parse_args()

    agent_dir = ROOT / 'agent_code' / args.agent
    if not agent_dir.is_dir():
        sys.exit(f'No such agent directory: {agent_dir}')

    name = args.name or args.agent
    files = list(collect_files(agent_dir))
    report = Report()

    print(f'\nChecking agent_code/{args.agent} ...')
    check_agent(agent_dir, files, report)
    req = check_requirements(report)
    if not args.no_check:
        play_one_game(args.agent, report)

    print(f'\nFiles to ship ({len(files)}):')
    for path in files:
        print(f'  {path.relative_to(agent_dir)}  ({path.stat().st_size / 1e3:.1f} kB)')

    if report.problems:
        print(f'\n{len(report.problems)} problem(s) -- not writing an archive.')
        sys.exit(1)

    out = build_zip(agent_dir, files, name, req, ROOT / args.out)
    print(f'\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB), '
          f'agent folder inside the zip: {name}/')
    if report.warnings:
        print(f'{len(report.warnings)} warning(s) above -- read them before uploading.')
    print('\nNext: run the Docker submission test (see README), then upload to MaMPF.')


if __name__ == '__main__':
    main()

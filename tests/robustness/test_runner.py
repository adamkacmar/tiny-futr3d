"""Exercise orchestration with a fake evaluator; never perform GPU inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Import before mocking print: Numba registers the original builtin at import.
import plugin.futr3d  # noqa: E402,F401

spec = importlib.util.spec_from_file_location(
    'robustness_eval', str(ROOT / 'tools/robustness_eval.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'results'
        self.addCleanup(self.temp.cleanup)

    def run_main(self, arguments, fake_evaluator=None):
        replacements = SimpleNamespace(
            STDOUT=subprocess.STDOUT,
            check_output=subprocess.check_output,
            run=fake_evaluator
            or mock.Mock(side_effect=AssertionError('Unexpected evaluation')))
        with mock.patch.object(sys, 'argv', ['robustness_eval.py'] + arguments), \
             mock.patch.object(runner, 'subprocess', replacements), \
             mock.patch.object(runner, 'digest', return_value='test-hash'):
            return runner.main()

    def test_preview_and_list_do_not_execute_or_create_outputs(self):
        with mock.patch('builtins.print'):
            self.assertEqual(self.run_main(['--list']), 0)
            self.assertEqual(
                self.run_main(
                    ['--all', '--dry-run', '--output-root',
                     str(self.output)]), 0)
        self.assertFalse(self.output.exists())

    def test_unique_directories_metrics_and_degradation(self):

        def fake(cmd, **kwargs):
            self.assertIn('--eval', cmd)
            self.assertEqual(cmd[cmd.index('--seed') + 1], '0')
            self.assertEqual(cmd[cmd.index('--gpu-id') + 1], '0')
            output = Path(cmd[cmd.index('--out') + 1]).parent
            metrics_dir = output / 'predictions/pts_bbox'
            metrics_dir.mkdir(parents=True)
            values = {
                'mean_ap': 0.7,
                'nd_score': 0.73
            } if output.name == 'clean' else {
                'mean_ap': 0.5,
                'nd_score': 0.6
            }
            (metrics_dir / 'metrics_summary.json').write_text(
                json.dumps(values))
            return SimpleNamespace(returncode=0)

        args = [
            '--scenario', 'clean', 'beam4', '--output-root',
            str(self.output)
        ]
        with mock.patch('builtins.print'):
            self.assertEqual(self.run_main(args, fake), 0)
        first = next(self.output.iterdir())
        snapshot = (first / 'results.json').read_bytes()
        result = json.loads(snapshot)
        self.assertAlmostEqual(result[1]['mAP_drop_pp'], 20)
        self.assertAlmostEqual(result[1]['NDS_drop_pp'], 13)
        self.assertEqual(result[1]['checkpoint_sha256'], 'test-hash')
        self.assertGreaterEqual(result[1]['runtime_seconds'], 0)
        self.assertTrue((first / 'beam4/config.py').is_file())
        with mock.patch('builtins.print'):
            self.assertEqual(self.run_main(args, fake), 0)
        self.assertEqual(len(list(self.output.iterdir())), 2)
        self.assertEqual((first / 'results.json').read_bytes(), snapshot)

    def test_failed_evaluation_stops_batch_and_records_failure(self):
        fake = mock.Mock(return_value=SimpleNamespace(returncode=42))
        with mock.patch('builtins.print'):
            code = self.run_main([
                '--scenario', 'clean', 'beam4', '--output-root',
                str(self.output)
            ], fake)
        self.assertEqual(code, 42)
        self.assertEqual(fake.call_count, 1)
        first = next(self.output.iterdir())
        records = json.loads((first / 'results.json').read_text())
        self.assertEqual(records[0]['returncode'], 42)
        self.assertIsNone(records[0]['NDS'])
        self.assertFalse((first / 'beam4').exists())


if __name__ == '__main__':
    unittest.main()

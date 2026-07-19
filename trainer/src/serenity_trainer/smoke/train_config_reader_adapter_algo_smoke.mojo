# serenity_trainer/smoke/train_config_reader_adapter_algo_smoke.mojo
#
# Guards Serenity-side adapter algorithm parsing against drift from
# serenitymojo.training.train_config. BOFT must remain intentionally excluded.

from std.testing import assert_equal, assert_raises, TestSuite

from serenity_trainer.util.config.TrainConfigReader import _adapter_algo_int


def test_adapter_algo_ids() raises:
    assert_equal(_adapter_algo_int(String("lora")), 0)
    assert_equal(_adapter_algo_int(String("loha")), 2)
    assert_equal(_adapter_algo_int(String("dora")), 3)
    assert_equal(_adapter_algo_int(String("lokr")), 4)
    assert_equal(_adapter_algo_int(String("oft")), 5)
    assert_equal(_adapter_algo_int(String("locon")), 7)
    assert_equal(_adapter_algo_int(String("lycoris")), 7)


def test_boft_rejected() raises:
    with assert_raises():
        _ = _adapter_algo_int(String("boft"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

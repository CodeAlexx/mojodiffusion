# io/samples_json_validate_smoke.mojo — validate serenity.sample_prompts.v1 files.
# Runs every path given on argv through the REAL reader
# (training/sample_prompt_config.mojo::read_sample_prompt_config) and asserts
# >=1 enabled prompt. Catches schema typos in authored samples JSONs before any
# trainer gate. CPU-only. Usage: samples_json_validate <file.json> [...]
from std.sys import argv

from serenitymojo.training.sample_prompt_config import read_sample_prompt_config


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: samples_json_validate <samples.json> [...]")
    var failures = 0
    for i in range(1, len(a)):
        var path = String(a[i])
        try:
            var cfg = read_sample_prompt_config(path)
            var enabled = 0
            for p in range(len(cfg.prompts)):
                if cfg.prompts[p].enabled:
                    enabled += 1
            if enabled == 0:
                print("FAIL (no enabled prompts):", path)
                failures += 1
            else:
                print("ok:", path, "-", enabled, "enabled prompt(s)")
        except e:
            print("FAIL (parse):", path, "-", String(e))
            failures += 1
    if failures > 0:
        raise Error(String(failures) + " samples JSON(s) failed validation")
    print("== ALL SAMPLES JSONS VALID ==")

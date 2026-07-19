# SDXL loader/saver/sampler/factory surface manifest.
#
# The previous monolithic gate hit the Mojo compiler OOM path under a 24 GB
# process cap on 2026-06-05. Keep this file tiny and run the split gates:
#   smoke/sdxl_surface_loader_contract_check.mojo
#   smoke/sdxl_surface_sampler_contract_check.mojo
#   smoke/sdxl_surface_saver_contract_check.mojo
#   smoke/sdxl_surface_factory_contract_check.mojo


def main():
    print("SDXL SURFACE MANIFEST OK")
    print("monolithic surface gate disabled after compiler OOM")
    print("run split gates: loader, sampler, saver, factory")

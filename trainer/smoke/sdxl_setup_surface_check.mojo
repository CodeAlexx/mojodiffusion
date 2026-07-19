# SDXL setup surface manifest.
#
# The previous monolithic assertion gate for SDXL setup/data-loader contracts
# triggered a Mojo compiler OOM on 2026-06-05: pid 58957 reached about 62 GiB
# RSS and was killed by the kernel, which also took down VSCode because the
# compiler process lived inside the VSCode snap scope.
#
# Keep this file intentionally tiny. Use the split gates instead:
#   smoke/sdxl_setup_base_contract_check.mojo
#   smoke/sdxl_setup_method_contract_check.mojo
#   smoke/sdxl_setup_dataloader_contract_check.mojo


def main():
    print("SDXL SETUP SURFACE MANIFEST OK")
    print("monolithic setup gate disabled after compiler OOM")
    print("run split gates: base, method, dataloader")

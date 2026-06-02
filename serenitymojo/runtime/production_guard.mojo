# production_guard.mojo - metadata checks for production-safe pipeline cleanup.
#
# This module does not inspect code automatically yet. It centralizes the policy
# language so adapters can later declare whether they still depend on debug
# host-readback paths.


@fieldwise_init
struct ProductionGuard(Copyable, Movable, ImplicitlyCopyable):
    var allows_host_tensor_readback: Bool
    var allows_host_activation_build: Bool
    var allows_cpu_tokenization: Bool
    var allows_cpu_scalar_schedule: Bool
    var allows_cpu_artifact_write: Bool

    def strict(self) -> Bool:
        return (
            not self.allows_host_tensor_readback
            and not self.allows_host_activation_build
            and self.allows_cpu_tokenization
            and self.allows_cpu_scalar_schedule
            and self.allows_cpu_artifact_write
        )


def production_gpu_math_guard() -> ProductionGuard:
    return ProductionGuard(False, False, True, True, True)


def debug_guard() -> ProductionGuard:
    return ProductionGuard(True, True, True, True, True)


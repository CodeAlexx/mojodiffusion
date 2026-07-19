# 1:1 port of Serenity modules/util/TrainProgress.py
#
# Tracks where a run is: epoch / epoch_step / epoch_sample / global_step.
# next_step / next_epoch / filename_string mirror the .py exactly.
# Host-only struct (no Tensor); Copyable so the driver can carry it by value.

@fieldwise_init
struct TrainProgress(Copyable, Movable):
    var epoch: Int
    var epoch_step: Int
    var epoch_sample: Int
    var global_step: Int

    # TrainProgress.__init__ defaults (TrainProgress.py:1-12): all zero.
    @staticmethod
    def zero() -> TrainProgress:
        return TrainProgress(0, 0, 0, 0)

    # next_step (TrainProgress.py:14-17): epoch_step += 1; epoch_sample += batch;
    # global_step += 1.
    def next_step(mut self, batch_size: Int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    # next_epoch (TrainProgress.py:19-22): reset epoch_step/epoch_sample, epoch += 1.
    def next_epoch(mut self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    # filename_string (TrainProgress.py:24-25): f"{global_step}-{epoch}-{epoch_step}".
    def filename_string(self) -> String:
        return (
            String(self.global_step)
            + "-"
            + String(self.epoch)
            + "-"
            + String(self.epoch_step)
        )

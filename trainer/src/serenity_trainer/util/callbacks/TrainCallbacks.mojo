"""Serenity-shaped train callback state.

Serenity wires TrainUI to the trainer through TrainCallbacks:
`on_update_train_progress(train_progress, max_step, max_epoch)` updates the UI
progress bars every step, and `on_update_status(status)` updates the status
label. Mojo cannot reuse the Python callable object directly, so this struct is
the same contract as retained state that UI/runtime bridges can copy from.
"""

from serenity_trainer.util.TrainProgress import TrainProgress


struct TrainCallbacks(Copyable, Movable):
    var progress: TrainProgress
    var max_step: Int
    var max_epoch: Int
    var status: String
    var sample_default_step: Int
    var sample_default_max_step: Int
    var sample_custom_step: Int
    var sample_custom_max_step: Int
    var sample_default_count: Int
    var sample_custom_count: Int

    def __init__(out self):
        self.progress = TrainProgress.zero()
        self.max_step = 0
        self.max_epoch = 0
        self.status = String("")
        self.sample_default_step = 0
        self.sample_default_max_step = 0
        self.sample_custom_step = 0
        self.sample_custom_max_step = 0
        self.sample_default_count = 0
        self.sample_custom_count = 0

    def on_update_train_progress(
        mut self,
        train_progress: TrainProgress,
        max_step: Int,
        max_epoch: Int,
    ):
        self.progress = train_progress.copy()
        self.max_step = max_step
        self.max_epoch = max_epoch

    def on_update_status(mut self, status: String):
        self.status = status.copy()

    def on_sample_default(mut self):
        self.sample_default_count += 1

    def on_update_sample_default_progress(mut self, step: Int, max_step: Int):
        self.sample_default_step = step
        self.sample_default_max_step = max_step

    def on_sample_custom(mut self):
        self.sample_custom_count += 1

    def on_update_sample_custom_progress(mut self, step: Int, max_step: Int):
        self.sample_custom_step = step
        self.sample_custom_max_step = max_step


def train_callback_progress_line(
    callbacks: TrainCallbacks,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    learning_rate: Float32,
) -> String:
    var status = callbacks.status.copy()
    if status.byte_length() == 0:
        status = String("Training ...")
    return train_callback_progress_line_values(
        callbacks.progress.epoch,
        callbacks.progress.epoch_step,
        callbacks.progress.global_step,
        callbacks.max_step,
        callbacks.max_epoch,
        loss,
        smooth_loss,
        grad_norm,
        learning_rate,
        status,
    )


def train_callback_progress_line_values(
    epoch: Int,
    epoch_step: Int,
    global_step: Int,
    max_step: Int,
    max_epoch: Int,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    learning_rate: Float32,
    status: String,
) -> String:
    var status_value = status.copy()
    if status_value.byte_length() == 0:
        status_value = String("Training ...")
    return (
        String("[Serenity-callback] progress epoch ")
        + String(epoch)
        + String("/")
        + String(max_epoch)
        + String(" | step ")
        + String(epoch_step)
        + String("/")
        + String(max_step)
        + String(" | global_step ")
        + String(global_step)
        + String(" | loss ")
        + String(loss)
        + String(" | smooth_loss ")
        + String(smooth_loss)
        + String(" | grad_norm ")
        + String(grad_norm)
        + String(" | lr ")
        + String(learning_rate)
        + String(" | status ")
        + status_value.copy()
    )


def append_train_callback_progress_line(
    path: String,
    callbacks: TrainCallbacks,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    learning_rate: Float32,
) raises:
    var f = open(path, "a")
    f.write(
        train_callback_progress_line(
            callbacks,
            loss,
            smooth_loss,
            grad_norm,
            learning_rate,
        )
    )
    f.write("\n")
    f.close()


def append_train_callback_progress_line_values(
    path: String,
    epoch: Int,
    epoch_step: Int,
    global_step: Int,
    max_step: Int,
    max_epoch: Int,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    learning_rate: Float32,
    status: String,
) raises:
    var f = open(path, "a")
    f.write(
        train_callback_progress_line_values(
            epoch,
            epoch_step,
            global_step,
            max_step,
            max_epoch,
            loss,
            smooth_loss,
            grad_norm,
            learning_rate,
            status,
        )
    )
    f.write("\n")
    f.close()

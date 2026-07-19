# BaseTrainer.mojo — 1:1 port of Serenity modules/trainer/BaseTrainer.py.
#
# Serenity's BaseTrainer is an ABC (TimedActionMixin + metaclass=ABCMeta) that
# holds the run config/callbacks/commands and exposes:
#   * abstract start()/train()/end()                     (BaseTrainer.py:36-46)
#   * create_model_loader / create_model_setup /
#     create_data_loader / create_model_saver /
#     create_model_sampler factories                     (BaseTrainer.py:48-83)
#   * _start_tensorboard / _stop_tensorboard             (BaseTrainer.py:84-103)
#
# PORT BOUNDARY: the create.* factories, the BaseModel*/Loader/Saver/Sampler
# hierarchies, and tensorboard/subprocess are out of this slice's scope (no
# Python, no data pipeline). Mojo has no ABC/metaclass, so the abstract surface
# becomes a `trait Trainer` (start/train/end), and the concrete state Serenity
# stores on `self` (config, train_device/temp_device) becomes a small struct
# `BaseTrainerState` that the concrete GenericTrainer driver composes.
#
# The tensorboard launch/kill and the create_* factories are kept as documented
# STUBS here: they belong to the harness/integration unit, not the math driver.

from serenity_trainer.util.config.TrainConfig import TrainConfig


# ── trait Trainer — the abstract surface (BaseTrainer.py:36-46) ───────────────
# Serenity marks start()/train()/end() @abstractmethod. Concrete trainers
# (GenericTrainer) implement train() as the epoch/step loop. Mojo dispatches the
# concrete spec by comptime monomorphization (cf. ModelSpec), so this trait
# documents the contract; the runnable loop is GenericTrainer.train (driver.mojo).
trait Trainer(Movable):
    # start() — one-time setup (load model, build data loader). Stubbed at the
    # port boundary (harness owns model/data construction).
    def start(mut self) raises:
        ...

    # train() — the epoch/step loop (GenericTrainer.train, ported in
    # trainer/GenericTrainer.mojo::train as a free function over a ModelSpec).
    def train(mut self) raises:
        ...

    # end() — teardown (save final, stop tensorboard).
    def end(mut self) raises:
        ...


# ── BaseTrainerState — the concrete state BaseTrainer.__init__ stores ─────────
# Mirrors BaseTrainer.py:28-34:
#   self.config = config
#   self.callbacks = callbacks      (harness-owned; out of scope here)
#   self.commands  = commands       (harness-owned; out of scope here)
#   self.train_device = torch.device(config.train_device)
#   self.temp_device  = torch.device(config.temp_device)
#
# Device handling: Serenity carries torch.device objects; in the Mojo port
# there is a single DeviceContext threaded through the driver, so the device
# fields collapse to the (boolean) "is the run on an accelerator" intent. We keep
# the config and a master-process flag (multi.is_master() guards in train()).
struct BaseTrainerState(Copyable, Movable):
    var config: TrainConfig
    var is_master: Bool        # multi.is_master() — single-process port → True

    def __init__(out self, config: TrainConfig):
        # super().__init__() (TimedActionMixin) — the timed-action cadence state
        # (last sample/backup/save times) is tracked by the driver's TimedAction
        # stubs (GenericTrainer needs_sample/needs_backup/needs_save), not here.
        self.config = config
        self.is_master = True


# ── create_* factories (BaseTrainer.py:48-83) — PORT-BOUNDARY STUBS ───────────
# Serenity dispatches on (model_type, training_method) via modules.util.create.
# Those builders construct the BaseModel/Loader/Saver/Sampler and the MGDS data
# loader — all out of scope for the pure-Mojo math driver (no Python, no MGDS).
# Kept as named no-ops so the BaseTrainer surface is structurally complete and the
# integration unit can fill them in against modules/util/create.py.
def create_model_loader() raises:
    # create.create_model_loader(model_type, training_method)  (BaseTrainer.py:48)
    pass


def create_model_setup() raises:
    # create.create_model_setup(...)                            (BaseTrainer.py:51)
    pass


def create_data_loader() raises:
    # create.create_data_loader(...)                            (BaseTrainer.py:60)
    # MGDS pipeline — explicitly out of scope (no Python, no MGDS).
    pass


def create_model_saver() raises:
    # create.create_model_saver(model_type, training_method)    (BaseTrainer.py:73)
    pass


def create_model_sampler() raises:
    # create.create_model_sampler(...)                          (BaseTrainer.py:76)
    pass


# ── tensorboard (BaseTrainer.py:84-103) — PORT-BOUNDARY STUBS ─────────────────
# _start_tensorboard spawns a `tensorboard` subprocess; _stop_tensorboard kills
# it. No subprocess/observability in the math slice — kept as documented no-ops.
def start_tensorboard() raises:
    # subprocess.Popen([tensorboard, --logdir, ...])            (BaseTrainer.py:84)
    pass


def stop_tensorboard() raises:
    # self.tensorboard_subprocess.kill()                        (BaseTrainer.py:102)
    pass

"""Serenity-shaped train command state.

Nerogar Serenity keeps UI actions out of the trainer loop with TrainCommands:
Stop is sticky, while sample/default, custom sample, backup, and save are
one-shot flags consumed by GenericTrainer.train(). This Mojo port keeps the same
semantics for the native UI bridge.
"""


struct TrainCommands(Movable):
    var stop_command: Bool
    var sample_custom_commands: List[String]
    var sample_default_command: Bool
    var backup_command: Bool
    var save_command: Bool

    def __init__(out self):
        self.stop_command = False
        self.sample_custom_commands = List[String]()
        self.sample_default_command = False
        self.backup_command = False
        self.save_command = False

    def reset(mut self):
        self.sample_custom_commands = List[String]()
        self.sample_default_command = False
        self.backup_command = False
        self.save_command = False

    def stop(mut self):
        self.stop_command = True

    def get_stop_command(self) -> Bool:
        return self.stop_command

    def sample_custom(mut self, sample_params_label: String):
        self.sample_custom_commands.append(sample_params_label.copy())

    def get_and_reset_sample_custom_commands(mut self) -> List[String]:
        var commands = self.sample_custom_commands.copy()
        self.sample_custom_commands = List[String]()
        return commands^

    def sample_default(mut self):
        self.sample_default_command = True

    def get_and_reset_sample_default_command(mut self) -> Bool:
        var command = self.sample_default_command
        self.sample_default_command = False
        return command

    def backup(mut self):
        self.backup_command = True

    def get_and_reset_backup_command(mut self) -> Bool:
        var command = self.backup_command
        self.backup_command = False
        return command

    def save(mut self):
        self.save_command = True

    def get_and_reset_save_command(mut self) -> Bool:
        var command = self.save_command
        self.save_command = False
        return command

    def merge(mut self, mut other: TrainCommands):
        if other.get_stop_command():
            self.stop()
        var custom = other.get_and_reset_sample_custom_commands()
        for i in range(len(custom)):
            self.sample_custom(custom[i].copy())
        if other.get_and_reset_sample_default_command():
            self.sample_default()
        if other.get_and_reset_backup_command():
            self.backup()
        if other.get_and_reset_save_command():
            self.save()

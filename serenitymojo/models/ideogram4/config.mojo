# Ideogram-4 model dims (migrated from serenity_trainer modelSampler/Ideogram4Sampler
# so serenitymojo/models/ideogram4/block.mojo resolves them without a serenity-trainer
# dependency — autograd_v2 engine adapter prerequisite, Stage 0).
comptime IDEOGRAM4_NUM_LAYERS = 34
comptime IDEOGRAM4_HIDDEN = 4608
comptime IDEOGRAM4_NUM_HEADS = 18
comptime IDEOGRAM4_HEAD_DIM = 256
comptime IDEOGRAM4_INTERMEDIATE_SIZE = 12288
comptime IDEOGRAM4_ADALN_DIM = 512

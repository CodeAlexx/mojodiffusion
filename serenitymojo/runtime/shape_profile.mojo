# shape_profile.mojo - compile-time specialization metadata.
#
# Shape profiles are the bridge between runtime requests and specialized Mojo
# entry points. They are metadata records only; model families still instantiate
# concrete comptime shapes in their adapters.


@fieldwise_init
struct ShapeProfile(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var width: Int
    var height: Int
    var frames: Int
    var latent_h: Int
    var latent_w: Int
    var latent_t: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var latent_channels: Int
    var patch_size: Int

    def is_video(self) -> Bool:
        return self.frames > 1

    def latent_cells(self) -> Int:
        return self.latent_t * self.latent_h * self.latent_w


def zimage_1024_profile() -> ShapeProfile:
    return ShapeProfile(
        String("zimage_1024"),
        1024,
        1024,
        1,
        128,
        128,
        1,
        4096,
        256,
        4352,
        16,
        2,
    )


def klein9b_1024_profile() -> ShapeProfile:
    return ShapeProfile(
        String("klein9b_1024"),
        1024,
        1024,
        1,
        64,
        64,
        1,
        4096,
        512,
        4608,
        128,
        1,
    )


def lance_tiny_video_profile() -> ShapeProfile:
    return ShapeProfile(
        String("lance_tiny_t3_h1_w1"),
        16,
        16,
        9,
        1,
        1,
        3,
        3,
        4,
        9,
        48,
        1,
    )


def lance_256_9f_profile() -> ShapeProfile:
    return ShapeProfile(
        String("lance_256_9f"),
        256,
        256,
        9,
        16,
        16,
        3,
        768,
        256,
        1028,
        48,
        1,
    )


# static_dispatch.mojo - metadata for finite Mojo specialization targets.
#
# Some model families cannot be selected by shape fully dynamically because
# their SDPA/mask paths are comptime-shaped. This file records the concrete
# specializations the production runner is allowed to dispatch to. It does not
# call model math; family adapters still instantiate the concrete types.


@fieldwise_init
struct StaticSpecialization(Copyable, Movable, ImplicitlyCopyable):
    var model_id: String
    var profile_name: String
    var entry_name: String
    var width: Int
    var height: Int
    var frames: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var pipeline_path: String

    def is_video(self) -> Bool:
        return self.frames > 1


def sensenova_u1_smoke_specialization() -> StaticSpecialization:
    return StaticSpecialization(
        String("sensenova_u1"),
        String("sensenova_u1_64_text18"),
        String("SenseNovaU1[4,18]"),
        64,
        64,
        1,
        4,
        18,
        22,
        String("serenitymojo/pipeline/sensenova_u1_gen_smoke.mojo"),
    )


def sensenova_u1_2048_text512_specialization() -> StaticSpecialization:
    return StaticSpecialization(
        String("sensenova_u1"),
        String("sensenova_u1_2048_text512"),
        String("SenseNovaU1[4096,512]"),
        2048,
        2048,
        1,
        4096,
        512,
        4608,
        String("serenitymojo/pipeline/sensenova_u1_pipeline.mojo"),
    )


def hidream_o1_smoke_specialization() -> StaticSpecialization:
    return StaticSpecialization(
        String("hidream_o1"),
        String("hidream_o1_64_s20"),
        String("HiDreamO1Offloaded[20]"),
        64,
        64,
        1,
        4,
        16,
        20,
        String("serenitymojo/pipeline/hidream_o1_smoke.mojo"),
    )


def hidream_o1_2048_text512_specialization() -> StaticSpecialization:
    return StaticSpecialization(
        String("hidream_o1"),
        String("hidream_o1_2048_s4608"),
        String("HiDreamO1Offloaded[4608]"),
        2048,
        2048,
        1,
        4096,
        512,
        4608,
        String("serenitymojo/pipeline/hidream_o1_pipeline.mojo"),
    )


def static_specialization_count() -> Int:
    return 4


def static_specialization_at(index: Int) raises -> StaticSpecialization:
    if index == 0:
        return sensenova_u1_smoke_specialization()
    if index == 1:
        return sensenova_u1_2048_text512_specialization()
    if index == 2:
        return hidream_o1_smoke_specialization()
    if index == 3:
        return hidream_o1_2048_text512_specialization()
    raise Error("static_specialization_at: index out of range")


def find_static_specialization(
    model_id: String, profile_name: String
) raises -> StaticSpecialization:
    for i in range(static_specialization_count()):
        var spec = static_specialization_at(i)
        if spec.model_id == model_id and spec.profile_name == profile_name:
            return spec
    raise Error(
        String("static specialization not registered: ")
        + model_id
        + String("/")
        + profile_name
    )

# request.mojo - user-facing generation request metadata.

from serenitymojo.runtime.model_manifest import ModelFamily


@fieldwise_init
struct GenerationRequest(Movable):
    var model_id: String
    var family: ModelFamily
    var prompt: String
    var negative_prompt: String
    var width: Int
    var height: Int
    var frames: Int
    var steps: Int
    var seed: UInt64
    var guidance_scale: Float32
    var output_path: String

    def is_video(self) -> Bool:
        return (
            self.family == ModelFamily.text_to_video()
            or self.family == ModelFamily.video_to_video()
            or self.frames > 1
        )


def default_t2i_request(model_id: String, prompt: String, output_path: String) -> GenerationRequest:
    return GenerationRequest(
        model_id,
        ModelFamily.text_to_image(),
        prompt,
        String(""),
        1024,
        1024,
        1,
        20,
        UInt64(42),
        Float32(4.0),
        output_path,
    )


def default_t2v_request(model_id: String, prompt: String, output_path: String) -> GenerationRequest:
    return GenerationRequest(
        model_id,
        ModelFamily.text_to_video(),
        prompt,
        String(""),
        256,
        256,
        9,
        20,
        UInt64(42),
        Float32(4.0),
        output_path,
    )


# LTX2 in Serenity Canvas

This guide covers the LTX2 controls currently exposed in Serenity Canvas,
including text-to-video, image-to-video, video-to-video, Retake, Extend,
Cinemagraph, Foley, LoRA loading, and the movable Canvas panels.

## Open Canvas

1. Start Serenity and open `http://127.0.0.1:7811`.
2. Hard-refresh once after a UI update so the browser loads the current
   JavaScript and CSS.
3. Open **Canvas**.
4. In the left parameters panel, choose **Create / canvas** for text-to-video
   or one of **I2V - LTX 2.3**, **Retake - LTX 2.3**, and
   **Extend - LTX 2.3**. The dedicated modes select the installed LTX 2.3
   checkpoint automatically.
5. Open the **LTX2 Mojo request** controls.

Choose **Native resolution** from the complete compiled LTX 2.3 matrix, then
enter a supported value in **Seconds**. Canvas automatically resolves the
corresponding native frame count and FPS; those implementation details remain
read-only so a request cannot be sent to an incompatible runner.

The native resolutions are Creator portrait, 540p landscape/portrait, 720p
landscape/portrait, and 1080p landscape/portrait. Seconds availability depends
on resolution: 540p reaches 20 seconds, 720p reaches 10 seconds, and 1080p is
currently compiled for 5 seconds.

**Quality** selects real transformer weight storage. The LTX Desktop creator
profile defaults to BF16:

- **BF16** uses the complete distilled LTX 2.3 checkpoint for the creator
  profile. Ordinary generation and I2V may also retain a user-selected,
  registry-classified LTX 2.3-compatible full finetune.
- **FP8** streams the native FP8 dev checkpoint with BF16 activations.
- **INT4** uses the resident W4A16 reconstruction path for lower memory.

Retake and Extend require BF16 plus a complete checkpoint because their source
video and source audio both pass through the bundled creator VAE encoders. The
partial diffusion-only FP8 files cannot implement that contract. Retake and
Extend therefore select the complete distilled checkpoint automatically.

For a first test:

- use a smaller native profile;
- leave **Post-upscale** set to **None**;
- leave the advanced Positive conditioning, Negative conditioning, and Noise
  fixture paths blank so the Mojo request runner builds the normal Gemma
  conditioning and noise;
- press **Load LTX Desktop creator profile** to select BF16, distilled
  eight-step Euler, automatic conditioning, generated audio, and no LoRA.

The activity line reports model loading, Gemma prompt encoding, denoising,
decoding, audio work, and post-processing.

## Text-to-video

1. Leave the Canvas source empty.
2. Set **LTX2 mode** to text-to-video, or leave automatic mode enabled.
3. Enter the video prompt.
4. Select a native profile.
5. Choose the audio policy.
6. Generate.

## Image-to-video

1. Set **Mode** to **I2V - LTX 2.3**.
2. Load one image into the source area, or select a gallery image and send it
   to Canvas.
3. Describe the motion and camera behavior in the prompt.
4. Select native resolution, duration, and precision. **Source preservation**
   starts at `1.00`.
5. Press **Generate I2V**.

The source image supplies frame-zero guidance and is encoded at both native
LTX stages. The native profile controls output dimensions, frame count, and
FPS; importing an image never replaces that profile with an arbitrary image
size. The mode rejects videos and refuses to generate until an image is loaded,
so it cannot silently fall back to text-to-video. Canvas uploads the original
selected file and gives that exact worker-readable path to LTX; painted masks
and the composited Canvas are never substituted for the selected I2V source.

## Video-to-video

1. Load a source video.
2. Set **LTX2 mode** to video-to-video.
3. Select the native profile matching the intended output.
4. Choose a V2V intent and describe the requested transformation or motion.
5. Generate.

For HEVC clips that Chrome cannot play, Canvas shows a generated poster while
submitting the original clip to the LTX2 worker.

The visible V2V intent buttons set **Source preservation**:

- **Replace subject (0.00)** gives the prompt full control for a total subject
  or character replacement.
- **Transform (0.30)** makes a strong edit but can still retain the original
  subject.
- **Preserve (0.70)** keeps most of the source subject and layout.

The slider remains available for manual values between these presets.

For masked video-to-video, add an **Inpaint Mask** layer from **Layers +**, use
**Mask Paint**, and paint the editable region white. Black preserves the source.

## Retake

1. Set **Mode** to **Retake - LTX 2.3**.
2. Load a source video.
3. Wait for Canvas to report its probed resolution, frame count, and FPS.
4. Enter **Start** and **Duration**. Duration must be at least 2 seconds and the
   complete window must remain inside the source clip.
5. Describe what should happen during the selected window and press
   **Retake selected window**.

Retake requires a compiled native profile that exactly matches the source. It
encodes the complete source, fully regenerates only the selected temporal
region, and restores clean source latents after every denoising step outside
that region. A painted spatial mask cannot be combined with a Retake window.
When the source contains audio, the default audio policy preserves its original
audio stream.

## Extend

1. Set **Mode** to **Extend - LTX 2.3**.
2. Load a source video and choose **Before source** or **After source**.
3. Enter the desired added seconds.
4. Canvas selects the smallest longer compiled profile at the same native
   resolution and FPS, and shows the exact resulting extension.
5. Describe the new leading or trailing action and press **Extend video**.

Extend zero-pads the clean source video and audio latents at the selected edge,
regenerates the new region, and extends the temporal edit mask 0.5 seconds into
the kept source to blend the seam. The source-to-target frame delta is
validated as an 8-frame-aligned LTX duration before the GPU lease is acquired.
Generated audio is the default for Extend.

## Cinemagraph

1. Load one source image.
2. Select the **Cinemagraph** feature workflow.
3. Include the exact trigger `CINEMAGRAPH_MOTION` in the prompt.
4. Start with feature weight `0.9`.
5. Select a native video profile and generate.

The prompt should describe the small repeating motion while keeping the rest of
the scene stable.

## Foley audio

1. Load a source video whose resolution, frame count, and FPS exactly match the
   selected native profile.
2. Select the **Foley / V2A** feature workflow.
3. Set source strength to `1.00`.
4. Set audio policy to **Generate**.
5. Describe the intended environmental and action sounds, then generate.

Foley preserves the source video stream and generates audio for it. It does not
accept a painted mask.

## LoRA loaders

Choose a compatible LoRA and press **Add**. Each row has an enable checkbox,
model strength, and remove button.

**Clear all** removes every stored Canvas LoRA loader, including incompatible
loaders that are hidden after switching models. Starting a **New Canvas**
project also clears the complete LoRA list.

The **Load LTX Desktop creator profile** button clears all LoRA rows. Add an
overlay LoRA only when the selected full checkpoint and workflow require it;
the creator I2V, Retake, and Extend paths do not inject one.

## Move the side panels

Each Canvas side panel has a vertical edge handle:

- drag the left handle to slide the parameters panel partly or completely off
  the screen;
- drag the right handle to slide Layers and Gallery partly or completely off
  the screen;
- click a handle to hide or restore its panel;
- with a handle focused, use **Home** to restore, **End** to hide, and the arrow
  keys to move it in smaller steps;
- **Tab** toggles the Layers and Gallery panel.

Panel positions persist across refreshes. The center Canvas expands into the
space released by either panel.

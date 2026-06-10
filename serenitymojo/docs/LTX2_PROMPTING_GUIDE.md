# LTX-2 Prompting Guide (distilled)

- **Source**: https://ltx.io/blog/prompting-guide-for-ltx-2
- **Fetched**: 2026-06-09
- **Purpose**: Guides prompt construction for the pure-Mojo LTX-2 pipelines in this repo —
  `serenitymojo/pipeline/ltx2_t2v_av_mvp.mojo` (T2V + audio) and
  `serenitymojo/pipeline/ltx2_t2v_av_hq.mojo` (two-stage HQ).
- Content below is a faithful distillation of the LTX page; nothing here is invented.
  Where the guide is silent (negative prompts, parameter values), that is stated explicitly.

## Prompt structure

Write the prompt as a **single flowing paragraph** covering, in order:

1. **Shot establishment** — cinematography terms matching the genre, scale, and style.
2. **Scene setting** — lighting, color palette, textures, atmospheric mood.
3. **Action description** — the core sequence flowing naturally from beginning to end, in present tense.
4. **Character definition** — age, appearance, clothing, distinguishing details; convey emotion via physical cues.
5. **Camera movement** — when and how the view shifts; describe how the subject appears after the motion.
6. **Audio / dialogue** — ambient sounds, music descriptions, speech in quotation marks with language/accent notes.

## Length / level of detail

- **4–8 descriptive sentences** covering all key aspects.
- Match detail level to shot scale: closeups need precision; wide shots need less.
- Expect iteration — "refining your prompt is part of the workflow."

## Audio prompting (most relevant to our audio-quality work)

- "Audio features are translated into character movement, camera motion, and scene
  animation, producing coherent video sequences aligned to rhythm, energy, and timing."
- **Dialogue**: put the spoken text **in quotation marks**; optionally specify the
  "language and accent you would like the character to have."
  (e.g. `says with an angry african american accent: "..."`, `says with a low robotic voice: "..."`).
- **Ambient sound / music**: describe it in the scene text
  (e.g. "Ambient live music fills the space, led by her clear vocals over gentle acoustic strumming",
  "The faint hum of chatter and distant drilling fills the air").
- The guide lists "voice work in multiple languages" among things the model does well.

## What works well

- Cinematic compositions with thoughtful lighting.
- Emotive human moments and subtle facial nuance.
- Atmospheric elements: fog, mist, golden hour, rain, reflections.
- Clear camera language ("slow dolly in," "handheld tracking").
- Stylized aesthetics named early in the prompt (noir, painterly, surreal).
- Voice work in multiple languages.

## What to avoid

- "Avoid emotional labels like 'sad' or 'confused' without describing visual cues.
  Use posture, gesture, and facial expression instead."
- "LTX-2 does not currently generate readable or consistent text. Avoid signage,
  brand names, or printed material."
- Complex physics (jumping, juggling cause artifacts).
- Scene overload — too many characters/objects reduce accuracy.
- Conflicting light sources without motivation.
- Overcomplicated prompts — the more elements packed in, the "higher the chance some won't be seen."

## Negative prompts

The guide gives **no negative-prompt guidance**; the "what to avoid" items above are
about positive-prompt content, not a negative-prompt string.

## Parameters

The guide ties **no explicit parameters** (guidance scale, resolution, duration) to
prompting.

## Verbatim examples (copied from the guide)

### Dialogue + camera reveal (family garden scene)

> "A warm sunny backyard. The camera starts in a tight cinematic close-up of a woman and a man in their 30s, facing each other with serious expressions. The woman, emotional and dramatic, says softly, "That's it... Dad's lost it. And we've lost Dad." The man exhales, slightly annoyed: "Stop being so dramatic, Jess." A beat. He glances aside, then mutters defensively, "He's just having fun." The camera slowly pans right, revealing the grandfather in the garden wearing enormous butterfly wings, waving his arms in the air like he's trying to take off. He shouts, "Wheeeew!" as he flaps his wings with full commitment. The woman covers her face, on the verge of tears. The tone is deadpan, absurd, and quietly tragic."

### Voice/accent specification (sci-fi scene)

> "The young african american woman wearing a futuristic transparent visor and a bodysuit with a tube attached to her neck. she is soldering a robotic arm. she stops and looks to her right as she hears a suspicious strong hit sound from a distance. she gets up slowly from her chair and says with an angry african american accent: "Rick I told you to close that goddamn door after you!". then, a futuristic blue alien explorer with dreadlocks wearing a rugged outfit walks into the scene excitedly holding a futuristic device and says with a low robotic voice: "Fuck the door look what I found!". the alien hands the woman the device, she looks down at it excitedly as the camera zooms in on her intrigued illuminated face. she then says: "is this what I think it is?" she smiles excitedly. sci-fi style cinematic scene"

### Music + ambient audio (bar performance)

> "A warm, intimate cinematic performance inside a cozy, wood-paneled bar, lit with soft amber practical lights and shallow depth of field that creates glowing bokeh in the background. The shot opens in a medium close-up on a young female singer in her 20s with short brown hair and bangs, singing into a microphone while strumming an acoustic guitar, her eyes closed and posture relaxed. The camera slowly arcs left around her, keeping her face and mic in sharp focus as two male band members playing guitars remain softly blurred behind her. Warm light wraps around her face and hair as framed photos and wooden walls drift past in the background. Ambient live music fills the space, led by her clear vocals over gentle acoustic strumming."

## Quick checklist for our T2V+audio prompts

- [ ] Single paragraph, 4–8 sentences, present tense.
- [ ] Shot/style named early; lighting + atmosphere described.
- [ ] Emotion shown via posture/gesture/expression, never bare labels.
- [ ] Dialogue in quotation marks; voice/accent/language stated when it matters.
- [ ] Ambient sound or music described explicitly (this is what exercises the audio path).
- [ ] No on-screen text/signage, no complex physics, no scene overload.

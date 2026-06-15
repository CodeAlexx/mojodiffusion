# pipeline/ideogram4_magic.mojo — pure-Mojo magic prompt: plain text -> JSON caption.
# Runs Qwen3-8B (lm_head present) autoregressively in Mojo with a magic-prompt
# system prompt, emitting an Ideogram-4 structured JSON caption. The JSON is then
# tokenized + fed to ideogram4_generate (separate step). No external LLM/API.
from std.gpu.host import DeviceContext
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.text_encoder.qwen3_magic import generate_greedy
from json.parser import loads
from json.canonical import minify
from serenitymojo.io.ffi import BytePtr

comptime QWEN = "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
comptime TOKJSON = QWEN + "/tokenizer.json"
comptime EOS = 151645      # <|im_end|>
comptime PAD = 151643      # <|endoftext|>


def _system_prompt() -> String:
    # Ported verbatim from the KJ/Comfy "Ideogram4 Caption Prompt Template"
    # ([SYSTEM] section) the user uses in SwarmUI/ComfyUI. Slim single-shot
    # magic prompt: natural-language idea -> structured JSON caption
    # (aspect_ratio / high_level_description / compositional_deconstruction).
    return String(
        "You convert a natural-language user idea into a structured JSON caption an image renderer can consume. You receive the user idea plus a target aspect ratio, and you emit one JSON object.\n"
        "\n"
        "## OUTPUT CONTRACT — exactly three top-level keys, in this order:\n"
        "\n"
        "```json\n"
        "{\"aspect_ratio\":\"W:H\",\"high_level_description\":\"...\",\"compositional_deconstruction\":{\"background\":\"...\",\"elements\":[ ... ]}}\n"
        "```\n"
        "\n"
        "- Emit a SINGLE-LINE MINIFIED JSON object — no markdown fences, no commentary, no other top-level keys.\n"
        "- Preserve non-ASCII characters as-is (CJK, Cyrillic, Devanagari, Arabic, accented Latin). Never escape with `\\uNNNN`, transliterate, or replace `café` with `cafe`.\n"
        "- Use SINGLE quotes for embedded text references in prose fields (`'Joe's Diner'`, not `\\\"Joe's Diner\\\"`). The `text` field of text elements is the exception — that field holds the user's verbatim characters, may use any characters, and follows QUOTED SPAN FIDELITY below.\n"
        "\n"
        "### `aspect_ratio` (first field, always required)\n"
        "\n"
        "A string in `W:H` form with positive integers (`1:1`, `16:9`, `9:16`, `4:5`, `3:1`, `2:3`, etc.).\n"
        "- If the user message gives a concrete `W:H`, echo it verbatim.\n"
        "- If the user message says `auto`, pick a concrete ratio that matches the medium and composition (panoramic subjects → wide ratios like `16:9` or `3:1`; portrait subjects → tall like `9:16` or `4:5`; designed artifacts → format conventions like `2:3` book cover, `3:4` poster; ambiguous → `1:1`). NEVER emit the literal string `auto`.\n"
        "- The aspect ratio you commit to drives every bbox decision. Pick it first.\n"
        "\n"
        "### `high_level_description` — observational summary (50-word hard cap)\n"
        "\n"
        "- ONE long sentence preferred, never more than two.\n"
        "- Reads like a short natural-language prompt, not an analysis. Starts immediately with the subject — no \"this image shows\", \"depicts\", \"captures\".\n"
        "- Identifies subject(s), medium, and overall composition. Names recognized pop-culture entities by full name (`Nike Air Jordan 1`, `Eiffel Tower`, `Mario (Nintendo character)`).\n"
        "- Don't enumerate granular features (every color, every grid dimension, every typography choice). That detail belongs in element descs or `background`.\n"
        "- `various`, `multiple`, general categories ARE appropriate here. Specificity rule (below) applies to element descs and `background`, NOT this field.\n"
        "- For transparent backgrounds, include the literal phrase `on a transparent background`.\n"
        "\n"
        "GOOD: `A full-action shot of a male soccer player in a red kit and black Adidas cleats kicking a soccer ball on a green turf field, with a blurred crowd in the stadium background.`\n"
        "BAD (over-specifies): `A male soccer player captured mid-kick on a bright green grass pitch, right leg fully extended through the follow-through at the precise moment his black-and-white studded boot makes contact with a white-and-black size-5 ball...`\n"
        "\n"
        "## ELEMENTS — what they are, what they're not\n"
        "\n"
        "Each element is one of:\n"
        "```\n"
        "{\"type\":\"obj\",\"bbox\":[y1,x1,y2,x2],\"desc\":\"...\"}\n"
        "{\"type\":\"text\",\"bbox\":[y1,x1,y2,x2],\"text\":\"LINE ONE\\nLINE TWO\",\"desc\":\"...\"}\n"
        "```\n"
        "\n"
        "`bbox` is optional per-element (see BBOX section below).\n"
        "\n"
        "### SINGLE SUBJECT = SINGLE ELEMENT\n"
        "\n"
        "A coherent subject — one animal, person, vehicle, building, plant, instrument, machine — is exactly ONE `obj` element. Anatomical and structural parts are descriptive attributes inside that element's `desc`, NOT separate elements.\n"
        "\n"
        "FORBIDDEN: a bee split into 8 elements (thorax/abdomen/wings/eyes/legs/...); a car split into 6 (body/wheels/windshield/...); a person split into 7 (head/torso/each limb/...); a building split into 5 (foundation/walls/windows/roof/door); a flower split into 3 (petals/stem/leaves).\n"
        "\n"
        "When MULTIPLE distinct subjects appear (a person AND a dog; two bees; three runners), use MULTIPLE elements — one per subject.\n"
        "\n"
        "**Test:** part-of-one-thing → goes in that thing's desc. Separate thing → its own element.\n"
        "\n"
        "**Transparent enclosure + featured contents = ONE element.** Display cases, snow globes, terrariums, aquariums, specimen jars, bell jars, vitrines containing a featured subject: name the enclosure + contents as a single unified desc.\n"
        "\n"
        "**Configured parts + revealed interior = ONE element.** A car with an open door, a machine with raised hood, a building with drawn curtains: the open state and any revealed interior are attributes of the single subject's desc, not separate elements.\n"
        "\n"
        "### Element desc — what to write (30–60 words, 60-word HARD CAP)\n"
        "\n"
        "Identity first, then major attributes briefly, then one distinguishing detail if relevant. Each desc is a standalone catalog entry — open with the subject's identity, not a referring phrase like \"the X\" that assumes the reader has seen the scene.\n"
        "\n"
        "GOOD (introduces from scratch):\n"
        "- `Woman walking on the platform, medium size. Shoulder-length dark wavy hair, medium skin tone, light blue button-down shirt and grey trousers. Small bag slung over the right shoulder.`\n"
        "- `Circular concrete tunnel entrance with glowing blue ring lights along the interior. Train tracks lead directly into the dark opening.`\n"
        "\n"
        "**Major attributes — always name:**\n"
        "- People: skin tone, hair (color + style), each visible garment with color, expression/gaze, pose, distinguishing feature (mole, glasses, jewelry, held prop).\n"
        "- Objects: shape, material, color, distinctive parts (handle, label, logo, marking).\n"
        "- Scenes/structures: type, primary material, color, distinctive structural elements.\n"
        "\n"
        "**Skip (eat word budget for marginal benefit):**\n"
        "- Surface-finish micro-prose (`finely granular matte texture with subtle sheen along the elytral ridges`). Pick one short descriptor (matte/glossy/metallic/textured) or omit.\n"
        "- Pose mechanics per-limb. Pick ONE summary action phrase plus the major attributes.\n"
        "- Camera/shadow/lighting micro-detail per element. Belongs in `background`.\n"
        "- Fabric weave, skin texture nuances, micro-anatomy.\n"
        "\n"
        "### Element desc — what NOT to include\n"
        "\n"
        "**No shadows.** Cast shadows, drop shadows, ground shadows, contact shadows, ambient occlusion — describe in `background` only when scene-wide, otherwise omit (the renderer infers them). Forbidden: `casts a thin hard shadow to the lower right`, `with a soft drop shadow beneath`.\n"
        "\n"
        "**No camera or render language.** Depth of field, focus, sharpness, bokeh, exposure, motion blur, lens flare, chromatic aberration, film grain — render properties belong in `high_level_description` or `background` as natural prose ONLY when the user prompt explicitly named them. NEVER inside an obj desc.\n"
        "  - EXCEPTION — viewpoint/angle (`from a low-angle perspective`, `bird's-eye view`, `eye-level`) IS allowed in obj descs when the prompt calls for it. Place once, usually in the focal subject's desc or background.\n"
        "\n"
        "**No describing impressions instead of physical reality.** Avoid `luminous`, `radiant`, `vibrant`, `lush`, `dynamic`, `glowing` (metaphorically), `gorgeous`, `stunning`, `breathtaking`, `mesmerizing`. Use observable properties: `cheekbone catches a small highlight`, not `luminous complexion`.\n"
        "\n"
        "**No scene-context repetition per-element.** Lighting direction, ambient surface, mounting context, weather → describe ONCE in `background`. Each element's desc focuses on what's UNIQUE to that element.\n"
        "\n"
        "### Anchor placements to named references\n"
        "\n"
        "Specify body parts, surfaces, spatial landmarks.\n"
        "- CORRECT: `applied to the forehead near the hairline above the left eyebrow`.\n"
        "- INCORRECT: `pressed against the skin`.\n"
        "- CORRECT: `resting on the lower-right corner of the table directly in front of the laptop`.\n"
        "- INCORRECT: `sitting on the surface`.\n"
        "\n"
        "## BACKGROUND — what goes here, what doesn't (CRITICAL)\n"
        "\n"
        "`background` describes the scene SHELL: walls and finishes, floor/ground and surface state, ceiling and architectural fixtures, windows as architecture, atmospheric context (sky, clouds, fog, dust, mist), scene-wide ambient lighting, distant out-of-focus context (horizon, blurred crowds, distant scenery).\n"
        "\n"
        "### No double-counting\n"
        "\n"
        "Anything described in `background` CANNOT also appear as an obj element. Each scene component lives in EXACTLY ONE field. Decide once and commit. Before emitting an obj element, scan `background` — if the component is named there, omit the obj element.\n"
        "\n"
        "### ALWAYS-BACKGROUND — these live in `background` only, never as obj elements:\n"
        "\n"
        "- sky, clouds, atmospheric color\n"
        "- horizon\n"
        "- distant mountains, hills, tree lines\n"
        "- atmospheric weather (fog, haze, mist, smoke)\n"
        "- distant cityscape or stadium architecture\n"
        "- distant blurred or simplified crowds\n"
        "- the floor / ground / turf / paving surface the scene sits on\n"
        "- ambient walls or studio backdrop behind focal subjects\n"
        "\n"
        "You cannot split these by region. `sky upper-left portion`, `sky behind the fortress`, `sky upper two-thirds` are the SAME component — describe in `background` once. Same for crowd, ground, horizon.\n"
        "\n"
        "If you want technique-level detail on an atmospheric component (watercolor wet-on-wet sky blooms, fog with directional density variation), put that detail in `background`. The `background` field is allowed to be long.\n"
        "\n"
        "### Ground/floor/pavement is ALWAYS background — zero tolerance\n"
        "\n"
        "The surface the scene sits on — floor, ground, turf, grass, dirt, sand, asphalt, pavement, road, sidewalk, deck, water surface, snow, tile floor, hardwood, marble — lives in `background` only. This holds REGARDLESS of how the input formats it: if the prompt lists `Wet rain-slicked pavement below` as a foreground bullet, RE-CLASSIFY it into background.\n"
        "\n"
        "**Surface character that belongs in background, not as a separate obj:** wet / rain-slicked / mud-streaked / dusty / cracked / polished / weathered surface state; reflective neon pools, fragmented color reflections, puddles, wet patches, mud patches, ice patches, frost, snow on the floor, water pooled on the ground, oil slicks, footprints, tire tracks; surface material (asphalt, cobblestone, hardwood, tile, marble, packed dirt); texture words for the floor (glassy, mirror-like, matte, polished, rough).\n"
        "\n"
        "**Puddles, reflections, wet patches are part of the ground surface** — never separate obj elements, regardless of whether they reflect the hero's silhouette or carry visible content.\n"
        "\n"
        "**Failure mode this prevents:** when a standing hero is the focal element and the floor is also emitted as an obj at the bottom of the frame, the renderer treats the floor obj as a 2D frame band rather than a perspectival receding plane, and clips the hero's legs into it — figure rendered half-in-the-ground with feet/calves buried.\n"
        "\n"
        "**Discrete objects ON the floor are still elements:** broken glass shards, crushed cans, scattered debris, leaves, rocks, dropped tools, brick fragments, foreground litter remain obj elements. The rule applies to the SURFACE itself and any state of that surface (wet, frozen, muddy, puddled), never to solid objects resting on it.\n"
        "\n"
        "### Background is the shell only — no individually-placeable things\n"
        "\n"
        "Furniture, vehicles, equipment, people, animals, decor (artwork, signs, plants in pots, stacks of books), free-standing lamps → obj elements, never `background`.\n"
        "\n"
        "### Shell-affixed prominent objects → DUAL MENTION\n"
        "\n"
        "Some objects are simultaneously part of the shell AND focal elements that define the room's identity: a chalkboard covering the back wall of a classroom, a fireplace built into a living-room wall, a large mounted TV, a stage proscenium, a built-in altar, a built-in bookshelf, a large fixed reception desk, a fixed sign/banner.\n"
        "\n"
        "For these, MANDATORY all three steps:\n"
        "1. **MENTION in `background`** as part of the shell — anchors the object to the wall.\n"
        "2. **EMIT as an obj element** with the qualifier `\"the primary background element\"` (or similar) at the start of its desc. The obj carries the detail (material, content, frame, mounting).\n"
        "3. **PLACE FIRST in the elements list** so painter's-algorithm draws it behind foreground items.\n"
        "\n"
        "Skipping step 1 (the most common failure) makes the renderer float the object in mid-room or render it in front of foreground subjects.\n"
        "\n"
        "This is an EXCEPTION to the shell rule's \"no individually placeable things\". Applies ONLY to objects that genuinely define the room's architectural identity. Free-standing items (chairs, table lamps, plants in pots, framed pictures on a wall) get the normal treatment: elements only, no background mention.\n"
        "\n"
        "### Recession/arrangement is not architecture\n"
        "\n"
        "Do not smuggle furniture or people into `background` by describing them as a receding arrangement. Forbidden background phrasings: `rows of desks recede toward the back`, `a grid of desks fills the room`, `students seated at the desks`, `chairs arranged in front of the podium`, `the room is filled with people`, `cars parked along the street`, `customers seated at the tables`. The arrangement IS the foreground content — emit elements.\n"
        "\n"
        "### No medium/post-processing effects in background\n"
        "\n"
        "`background` describes WHAT is in the scene, not HOW it was made. Forbidden in `background` — even when the prompt names the effect (route those to HLD instead):\n"
        "- Film grain, Kodak/Portra/Tri-X grain, ISO noise\n"
        "- Lens flare, chromatic aberration, vignetting, bokeh quality\n"
        "- Color cast / film-stock shift (warm shift, cool shift)\n"
        "- Paper texture, paper grain, canvas texture\n"
        "- Brushstroke texture, palette-knife texture\n"
        "- Halftone dots, screen-print texture, risograph texture\n"
        "\n"
        "**Test:** read `background` aloud. If you can picture the EMPTY room from the description — no furniture, no people, no equipment, no wall decor — you're in the shell. If anything disappears when you remove the room's contents, the background has leaked.\n"
        "\n"
        "## BBOX STRATEGY\n"
        "\n"
        "INCLUDE bboxes on elements where precise positioning matters — portrait subjects, products on a surface, logos, signs on a wall, distinct individually-placeable objects.\n"
        "\n"
        "OMIT bboxes on elements that represent dense or hard-to-enumerate visuals — crowds, fields of wildflowers, scattered particles, starry skies. Per-element judgment.\n"
        "\n"
        "### Coordinate system\n"
        "\n"
        "Coordinates are normalized to the target image shape: `x` runs left→right along full width (0 = left edge, 1000 = right), `y` runs top→bottom along full height (0 = top, 1000 = bottom). Top-left origin. Format `[y1, x1, y2, x2]` with `y1 < y2`, `x1 < x2`.\n"
        "\n"
        "### Shape warning (common failure)\n"
        "\n"
        "Bbox values are normalized to 0–1000 in BOTH axes. A square `[0, 0, 500, 500]` is square only on a square frame; on 16:9 it becomes a wide rectangle, on 9:16 a tall rectangle. Most bbox failures (extra subjects, duplicates, mis-scaled objects) come from this mismatch.\n"
        "\n"
        "For round objects or square on-screen regions, scale spans so `(x2-x1)/(y2-y1) ≈ W/H`. For single-subject prompts on wide frames, prefer narrower x-spans. For multi-subject prompts, give each a tight bbox so no one bbox dominates and invites a duplicate.\n"
        "\n"
        "## SPECIFICITY — commit to one value\n"
        "\n"
        "This JSON feeds a diffusion model. Leave nothing for the model to invent or choose.\n"
        "\n"
        "**Banned hedge phrasings** (in elements and background): `things like`, `such as`, `e.g.`, `for example`, `or similar`, `various`, `could include`, `might be`, `some kind of`, `style of`. Replace with concrete nouns, counts, colors, materials, poses.\n"
        "\n"
        "**Banned alternative listings for one property:** `pale institutional off-white or pale green`, `oak or walnut`, `cream or ivory`, `late afternoon or early evening`, `italic serif or italic sans-serif`, `bold or semibold`. Pick ONE and commit. `or` is reserved for the loader's exclusive-choice idiom (`'YES' or 'NO'`), not captioner hedging.\n"
        "\n"
        "**Typography specifically:** name ONE typeface category (serif OR sans-serif OR display OR script OR monospace), ONE weight (bold/regular/light/medium), ONE style (italic OR upright). Never two joined by `or`.\n"
        "\n"
        "**Banned \"implied/suggested\" hedges:** `a desk corner implied`, `a chair suggested beneath the figure`, `a building hinted at`, `a shadow that reads as a person`. If it's in the scene, paint it concretely. If it isn't, leave it out. Forbidden words: `implied, suggested, hinted, barely visible, possibly, perhaps, maybe, might be, could be, reads as, almost`.\n"
        "\n"
        "**Exhaustive content preservation.** When the user provides enumerable content — schedules, itineraries, lists, menu items, steps, names, times — every item must appear in the output. Use as many text elements as needed; never sacrifice completeness for layout.\n"
        "\n"
        "**Named prompt elements MUST appear.** Every explicitly-named visual unit in the user prompt MUST appear as its own element:\n"
        "- Input `text:` sections — every entry becomes its own text element, verbatim. Zero tolerance: 3 entries in input → ≥3 text elements in output. Empty `text: []` is the only case where text elements may be omitted on that basis.\n"
        "- Quoted strings (single or double quotes) — each is its own text element.\n"
        "- Speech bubbles / dialogue callouts / thought bubbles / captions — each gets a text element for the quoted string AND an obj element for the bubble/balloon/container.\n"
        "- Named decorative elements (`small medical cross icon top-left`, `airplane arc trajectory`, `flame-lick flourish at the tail`) — each gets its own obj.\n"
        "- Named badges / chips / CTAs / strips — each gets its own obj (and text if it carries a quoted string).\n"
        "- Named accents / graphic devices (`hairline rule`, `dot grid`, `accent line`, `divider`) — each gets its own obj UNLESS it's a scene-wide overlay belonging in `background`.\n"
        "\n"
        "**Test before emitting:** count named visual units in the user prompt; element list must contain at least that many.\n"
        "\n"
        "**No placeholder enumeration.** When the imagined image contains a sequentially-numbered, alphabetically-labeled, or otherwise individually-identified set (stones numbered 1–50, parking spaces A1–A20, place cards `1st`–`12th`, a periodic table of 118 elements, a calendar grid of 31 dates, a 22-name team roster), EACH item is its own element. No `etc.`, no `and so on`, no `6 through 49`, no single obj grouping all into one cluster. List ALL of them.\n"
        "\n"
        "The \"dense unenumerable group\" exception (crowd of thousands, field of wildflowers, starry sky) does NOT apply to enumerable sets — if items are sequentially identified, they're enumerable BY DEFINITION.\n"
        "\n"
        "**Don't invent visual concepts the user didn't ask for.** Forbidden without explicit user request: `glitch art`, `wireframe overlay`, `mesh that fragments the body`, `digital artifacts`, `dissolved`, `decompose`. If the prompt asks for a cinematic photo of a journalist, render a cinematic photo of a journalist — not a glitch-art composite.\n"
        "\n"
        "## PLANNING — turn the user idea into elements\n"
        "\n"
        "### 1. Pick a medium\n"
        "\n"
        "`photograph | illustration | 3D render | graphic design` — applies as natural-language framing inside HLD/background, NOT as a structured slot.\n"
        "\n"
        "Decision: **DESIGNED artifact vs CAPTURED / DRAWN / RENDERED moment.**\n"
        "- **graphic design** — poster, book cover, album cover, magazine cover, flyer, banner, social post, sticker, logo, wordmark, packaging, app icon, UI mockup, infographic, menu, greeting card, ticket, signage. If a human designer would sit at a desk to make it.\n"
        "- **photograph** — portrait, landscape, lifestyle, street, sport, wildlife, food, product, fashion editorial (when described as a photograph). Default for ambiguous everyday scenes.\n"
        "- **illustration** — cartoon, anime, manga, comic, watercolor, oil painting, ink, vector, pixel art, children's book illustration, named studios (Ghibli, KyoAni, Pixar 2D).\n"
        "- **3D render** — CGI, octane/unreal/blender, hyperrealistic product render, arch viz, isometric low-poly, voxel, named 3D studios.\n"
        "\n"
        "Silent / ambiguous → photograph (default). The subject's reality status does NOT override this default — wizards, dragons, aliens, robots in a photograph are valid; the brief must explicitly ASK for illustration / painting / render to get one.\n"
        "\n"
        "Imperative verbs at the start (\"Illustrate a…\", \"Paint a…\", \"Draw a…\", \"Render a…\") are NOT medium signals — they mean \"depict / show\". Default to photograph unless an explicit medium-noun or style name appears.\n"
        "\n"
        "### 2. Style commitment\n"
        "\n"
        "Inside HLD/background prose, name the style ONCE (`Studio Ghibli animation`, `Pixar 3D animation`, `35mm film photograph`, `iPhone photo`, `editorial digital painting`, `flat vector illustration`). Keep it short — recognizable style names are enough; the renderer knows them. Don't append technique detail (`with hand-painted gouache backgrounds`) on top of well-known names.\n"
        "\n"
        "**\"Professional picture/photo/portrait\" of a person means PROFESSIONAL CONTEXT, not professional camera equipment.** Read as corporate headshot, LinkedIn profile, business bio — neutral business attire, soft even daylight, neutral backdrop, friendly approachable expression. NOT dramatic studio rim-lighting, creamy DSLR bokeh, dark moody backdrop.\n"
        "\n"
        "### 3. Photoreal defaults — AVOID \"warm\"\n"
        "\n"
        "For photographic prompts (no specified medium beyond `photo`/`photorealistic`/`selfie`/real-world scene):\n"
        "- Default to iPhone aesthetic — phone snapshot, ambient natural light, neutral white balance, accurate (not flattering) skin tones, ordinary framing. AVOID DSLR-magazine markers (creamy bokeh, telephoto compression, dramatic rim lighting, cinematic grade) — those signal AI-generation.\n"
        "- Default lighting framing: `natural daylight`, `overcast daylight`, `diffused daylight`, `cool-neutral white balance`. The word **\"warm\"** (in any phrase: `warm light`, `warm window light`, `warm tone`, `warm grading`) is BANNED as a grading adjective — it triggers the amber/golden AI look that ruins photorealism. When a scene physically has a warm-coloured light source (candle, sodium streetlamp, sunset), describe the SOURCE concretely (`candle flame`, `sodium streetlamp`) and the colour of the LIGHT POOL (`amber pool from the candle`) — but the global grade stays neutral.\n"
        "- Default composition: prefer non-centered framing (off-center, rule-of-thirds, asymmetrical, leading lines) for portraits, products, single-subject scenes. Use centered framing ONLY when the prompt explicitly calls for it (`centered`, `symmetrical`, `mandala`, `kaleidoscope`) or when the genre is inherently symmetric.\n"
        "- No motion blur in candid/realistic/iPhone-aesthetic photos. Motion blur is a craft signature (long-exposure pans, light streaks); using it in a candid signals AI. Real phone snapshots freeze the moment.\n"
        "- Saturation: don't stack `vibrant + bright + intense + saturated + electric + neon` for a neutral subject. Mention saturation ONCE (in HLD or background) only when the prompt explicitly asks.\n"
        "\n"
        "### 4. Populate underspecified scenes\n"
        "\n"
        "When the brief is sparse, don't render only what's explicitly named. Real scenes are populated. Add believable secondary subjects, micro-props that imply the subject's life, environmental texture, small narrative moments. Each invented element should belong in the world the brief implies — a paddy-field food stall plausibly has a chicken, a sauce bowl, a hand-painted price sign, a lantern.\n"
        "\n"
        "**Populate by depth layer.** Foreground (often-skipped), midground, background — each gets its own content. A foreground crop (an out-of-focus leaf at the bottom corner, the rim of a bowl, a fly mid-air close to camera) separates a real photograph from a postcard.\n"
        "\n"
        "**Commit to a specific cultural / regional identity.** \"Southeast Asian village\" is a hedge that produces generic AI visuals. \"Vietnamese pho stall by the rice paddies outside Hoi An\" is a real place. Specific commitment shapes architecture, signage script, food, dress, props.\n"
        "\n"
        "**Built environments need text everywhere.** Real shops, stalls, restaurants, vehicles, signage carry text on practically every surface. Generate text generously: shop name sign, sub-signs (`OPEN` / `TODAY'S SPECIAL`), menu board with handwritten items, price labels, jar/bottle labels, name tags, posters, fortune slips, vehicle/equipment labels, sponsor logos. `text: []` is almost always wrong for built environments — if your scene has a shop/stall/restaurant/workshop/market/vehicle, populate text. Specific content, never `various labels` or `menu items`.\n"
        "\n"
        "**Override:** when the brief explicitly says `minimal`, `sparse`, `empty`, `lonely`, `isolated`, `quiet`, `still`, `negative space`, `alone`, `single subject`, `in the middle of nowhere`, respect the restraint and skip populate.\n"
        "\n"
        "**Fantastical / sci-fi / fantasy / futuristic briefs get a populate bonus.** Stack sky drama (galaxies, ringed planets, multiple moons, nebulae), opposing focal points (volcano right / waterfall left), mid-distance scale anchors (crystal columns, futuristic cityscape, megastructures), light/energy effects throughout, exotic architecture/geology, deeply saturated palettes.\n"
        "\n"
        "## TEXT HANDLING\n"
        "\n"
        "For each text element:\n"
        "- `text` — literal characters appearing in the image, verbatim. Preserve diacritics, capitalization, punctuation. Never transliterate or strip.\n"
        "- `bbox` — optional, same coordinate system as obj elements.\n"
        "- `desc` — free-form prose covering size, location, font style, color, orientation, visual effects.\n"
        "\n"
        "**Sources of text to include:**\n"
        "1. **User-quoted text** (single OR double quotes) — verbatim, exact characters.\n"
        "2. **Format-required text** — headlines, taglines, author names, dates, venues, CTA copy, brand names, publisher marks, edition numbers (when format implies them).\n"
        "3. **In-scene contextual text** — signage, labels, license plates, badges, jersey numbers, t-shirt prints, awnings, neon signs, name tags.\n"
        "4. **Numeric content** — race numbers, jersey numbers, dates, prices, scores, time displays, address numbers. Numbers ARE text.\n"
        "5. **Prominent product brand text** — if an element names a prominent product (bottle, cosmetic, package, beverage) and the user didn't supply a real brand, invent a complete brand identity and list every label as text elements.\n"
        "\n"
        "**Rules:**\n"
        "- Exhaustive: if a viewer could read it, it goes in the list.\n"
        "- Each text element appears ONCE in the list. Do NOT also describe its characters in `description` — refer by role/position instead.\n"
        "- Use `\\n` for line breaks WITHIN a single text element (multi-line sign, stacked headline). Use SEPARATE list items for visually distinct text blocks.\n"
        "- For stylized hero typography where each letter is a distinct visual unit, stack with `\\n` at natural word breaks — long single-line stylized titles produce typos and dropped letters. e.g., `\"ENTRE\\nVERSOS E\\nCONTOS\"` not `\"ENTRE VERSOS E CONTOS\"`.\n"
        "- **Language scoping:** `scene`/`elements`/`description`/position descriptors are always in ENGLISH regardless of the user's brief language. Only the literal `text` field characters follow the user's brief language. Portuguese brief → English prose + Portuguese `text:` content.\n"
        "\n"
        "## POP CULTURE, BRANDS, NAMED REFERENCES\n"
        "\n"
        "When the user idea names or clearly implies a brand, trademark, product (sneaker/car/device), public figure, athlete, musician, actor, fictional character, film, show, game, franchise, team — the output MUST carry an explicit named reference in the relevant element `desc`, not a generic stand-in describing the look.\n"
        "\n"
        "Don't replace `Nike Dunk Low Panda` with `black and white retro sneakers`, `Spider-Man` with `a red-and-blue masked superhero`, `The Beatles` with `four men in matching suits` — unless the user asked for an anonymous lookalike. Name the specific thing the user pointed at.\n"
        "\n"
        "## TRANSPARENT BACKGROUND\n"
        "\n"
        "If the user's idea calls for transparent background, transparent canvas, alpha channel, cutout/isolated subject, sticker-style with no backdrop, or similar, the `background` field MUST be exactly this string, verbatim and nothing else: `transparent background`\n"
        "\n"
        "Do not paraphrase (no `clear backdrop`, `empty alpha`, `no background`, `PNG transparency`).\n"
        "\n"
        "In `high_level_description`, include the literal phrase `on a transparent background`."
    )


def _byte_substr(s: String, start: Int, end: Int) -> String:
    """Byte substring s[start:end). Mojo String has no slice operator, so view the
    backing bytes and copy out the range. (Mirrors serve/proc_ipc.byte_substr.)"""
    var n = end - start
    if n <= 0:
        return String("")
    var sp = s.as_bytes()
    var base = BytePtr(unsafe_from_address=Int(sp.unsafe_ptr()) + start)
    return String(StringSlice(ptr=base, length=n))


def _strip_trailing_commas(s: String) raises -> String:
    """Remove any ',' immediately followed (ignoring whitespace) by '}' or ']'.
    LLMs emit these trailing commas; json.parser.loads (strict RFC8259) rejects
    them. A comma is dropped ONLY when, scanning forward past spaces/tabs/newlines/
    carriage-returns, the next non-space byte is '}' or ']'. Commas inside double-
    quoted string literals are never touched: we track whether we are inside a
    string and honor backslash escapes."""
    var bs = s.as_bytes()
    var n = s.byte_length()
    var out = List[UInt8]()
    var in_str = False
    var escaped = False
    for i in range(n):
        var c = bs[i]
        if in_str:
            out.append(c)
            if escaped:
                escaped = False
            elif c == 0x5C:        # backslash -> next char is escaped
                escaped = True
            elif c == 0x22:        # closing quote
                in_str = False
            continue
        # not inside a string literal
        if c == 0x22:              # opening quote
            in_str = True
            out.append(c)
            continue
        if c == 0x2C:              # ',' — look ahead past whitespace
            var j = i + 1
            while j < n:
                var d = bs[j]
                if d == 0x20 or d == 0x09 or d == 0x0A or d == 0x0D:
                    j += 1
                    continue
                break
            if j < n and (bs[j] == 0x7D or bs[j] == 0x5D):  # '}' or ']'
                # drop this comma (do not append); whitespace is emitted as normal
                # by subsequent iterations
                continue
            out.append(c)
            continue
        out.append(c)
    return String(from_utf8=out)


def _normalize_caption(raw: String) raises -> String:
    """Turn a raw LLM completion into CANONICAL minified Ideogram-4 JSON.

    1. Trim. If a markdown code fence (```) is present, take the content between the
       first fence pair and strip a leading `json` language tag.
    2. Slice from the FIRST '{' to the LAST '}' (the JSON object), discarding any
       LLM preamble/suffix.
    3. Strip trailing commas (LLMs emit them; strict loads rejects them).
    4. loads() -> minify() (INSERTION-ORDER keys, preserving trained key order).
       If loads raises, re-raise with a short prefix of the raw input (FAIL LOUD —
       never return un-parseable JSON)."""
    var s = String(raw.strip())

    # 1. markdown code fence extraction
    var fence = String("```")
    var f0 = s.find(fence)
    if f0 >= 0:
        var inner_start = f0 + 3
        var f1 = s.find(fence, inner_start)
        if f1 > inner_start:
            s = String(_byte_substr(s, inner_start, f1).strip())
        else:
            # opening fence only; take everything after it
            s = String(_byte_substr(s, inner_start, s.byte_length()).strip())
        # strip a leading `json` language tag
        if s.find("json") == 0:
            s = String(_byte_substr(s, 4, s.byte_length()).strip())

    # 2. slice FIRST '{' .. LAST '}'
    var lb = s.find("{")
    var rb = s.rfind("}")
    if lb < 0 or rb < 0 or rb < lb:
        raise Error(
            String("magic_expand: no JSON object found in LLM output; prefix=")
            + _byte_substr(String(raw.strip()), 0, 200)
        )
    s = _byte_substr(s, lb, rb + 1)

    # 3. strip trailing commas
    s = _strip_trailing_commas(s)

    # 4. parse (strict) then canonical minify (insertion order)
    try:
        var v = loads(s)
        return minify(v)
    except e:
        raise Error(
            String("magic_expand: LLM emitted un-parseable JSON (")
            + String(e)
            + String("); cleaned prefix=")
            + _byte_substr(s, 0, 200)
        )


def magic_expand(plain: String, aspect: String, ctx: DeviceContext) raises -> String:
    """Expand a plain prompt into a CANONICAL minified Ideogram-4 JSON caption.

    Builds the chat (magic-prompt system + user line), encodes with the Qwen3
    tokenizer, loads Qwen3-8B, runs greedy generation, decodes, and normalizes the
    result to canonical minified JSON. Loud-fails if the model output is not a
    parseable JSON object."""
    var tok = Qwen3Tokenizer(TOKJSON)
    var user = (
        String("TARGET IMAGE ASPECT RATIO: ") + aspect
        + String(" (width:height).\nUser idea: ") + plain + String(" /no_think")
    )
    var chat = (
        String("<|im_start|>system\n") + _system_prompt() + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids = tok.encode(chat)
    var qwen = Qwen3Encoder.load(QWEN, Qwen3Config.klein_9b(), ctx)
    var gen = generate_greedy(qwen, ids, 1700, EOS, PAD, 2048, ctx)
    var decoded = tok.decode(gen)
    return _normalize_caption(decoded)


def main() raises:
    var ctx = DeviceContext()
    var plain = String("a red cube on a white table")
    var aspect = String("1:1")
    print("expanding magic prompt (greedy)...")
    var caption = magic_expand(plain, aspect, ctx)
    print("=== MAGIC PROMPT JSON ===")
    print(caption)
    print("=== END ===")

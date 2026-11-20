# Amazon Ad Agent

Builds faceless product videos (MP4, **always under 9 minutes**) from the photos and clips *you* supply, with a natural script, voice-over,
five visual templates, and a review checklist. It runs on GitHub Actions, **never uploads anything**, and you review and post each video yourself.

> **No Amazon scraping.** Amazon's Associates licence prohibits bots/data-extraction tools, and without the Creators/PA API you may not pull
> its images, ratings, reviews or prices. So the agent uses footage and facts you put in the project folders.

## Three video types

| Type | Folder holds | Length (auto) | What you get |
|---|---|---|---|
| **Review** (one product) | `facts.yaml` + `assets/` | about 5 min (3.5-8) | hook, overview, features, in use, limits, who it suits, verdict |
| **Comparison** (2-3 products) | `compare.yaml` | about 7.5 min (5.5-8.8) | a section per product, head-to-head on points every product has facts for, who each suits |
| **Top-N list** (3-9 products) | `list.yaml` | about 7.5 min (5.5-8.8) | countdown with a big **#N** badge per item, "best for" labels, recap |

`length: short` in the workflow gives 1.5-4.8 minutes for any type. All presets are in `config.yaml` (`length_presets`).
**Max 2 videos per day** (`safety.max_videos_per_day`), a brake against mass-production.

## Five visual templates (workflow dropdown `template`, or `video.template` in `config.yaml`)

| Key | Look |
|---|---|
| `one` | Clean and minimal: white panels, blue accent, soft fades |
| `two` | Bold and vibrant: solid colour captions, vivid backdrops, push transitions |
| `three` | Dark cinematic: letterbox bars, film grain, outlined captions, dip-to-black |
| `four` | Modern glass: real frosted-glass panels, pastel bokeh |
| `best` | Modern default: dark pill captions, amber accent |

Run **mode: preview** to get `template_preview.png`, one sample picture per template on one of your own photos. Templates are plain settings in
`config.yaml` (`templates:`), so you can tweak colours, caption style, grade, letterbox and transitions.

## Where things go

```
projects/<ASIN>/            a product:  facts.yaml, assets/ (photos + clips), music/ (optional), script.md
projects/<NAME>/            a comparison: compare.yaml  (or a list: list.yaml), script.md
projects/_template*/        blank copies to start from
```
**Keywords:** add `keywords:` to `facts.yaml`, `compare.yaml` or `list.yaml`. They are worked naturally into the title, description and tags (never stuffed).

## Making a video

1. Product folder: copy `projects/_template/` to `projects/<ASIN>/`, fill `facts.yaml` (true notes; **3+ hands_on_notes** from real use), add 8+ photos and some clips to `assets/`.
2. For a comparison or list: copy `projects/_template_compare/` or `projects/_template_list/` to `projects/<NAME>/` and fill it in.
3. **Actions -> Make video -> Run workflow**: project name, **mode: script** (tick **force** only to overwrite an existing script). Edit `projects/<NAME>/script.md`.
4. Optional: **mode: preview** to compare the five templates.
5. **mode: render** with `preview_seconds: 60` and an `edge_*` voice first; then again with `0`. Download the zip from the run's **Artifacts**:
   the `.mp4`, `title.txt`, `description.txt`, `pinned_comment.txt` and `REVIEW_CHECKLIST.md`.
6. Watch the whole video with sound before posting.

You can also write your own `script.md` (one line per sentence: `[id] text | product: k | callout: ... | assets: ...`) and just render.

## AI ad clips (optional, free GPU, untested on a real GPU)

**Make video -> mode: scenes** plans `scenes.md` for one product: polished commercial-style clips animated from your own product photos
(slow orbit/push-in, light sweep, rim light; up to `scenes.max_product_scenes`, product scenes first) plus faceless mood scenes. Unsafe prompts
(faces, frost/steam/instant effects, brand names, prices) are dropped. Then run `notebooks/make_ai_clips.ipynb` in Kaggle or Colab; it saves
`ai_*.mp4` into `assets/`. Renders use them automatically, add an on-screen "AI-generated scene" tag, an AI note in the description, and a
reminder to answer YES to YouTube's synthetic-content question. Your own video clips are graded and styled by the template but are not
AI-converted. Always check AI clips for distorted products or hands.

## AliExpress products (alongside Amazon)
Any video (single, comparison, list) can use AliExpress products, Amazon products, or both. Copy `projects/_template_aliexpress/` to
`projects/<ITEM ID>/` (the number in the AliExpress product URL), set `store: aliexpress` and your AliExpress affiliate link; everything else
works as for Amazon. As with Amazon, the agent never reads the shop: you supply your own photos, clips and true facts (listing photos belong to
the sellers). The Amazon Associates sentence is used ONLY when an Amazon product is in the video; AliExpress-only videos get a general affiliate
disclosure; mixed videos get both, and the description tags each link `[Amazon]` / `[AliExpress]`. Prices, shipping, delivery times, seller ratings
and order counts are never said. For comparison/list videos list the folders in `compare.yaml` / `list.yaml` exactly as before.
Check your Amazon Associates and AliExpress affiliate rules for videos that feature both shops: I could not verify them.

## Footage selection (upload as many photos and clips as you like)
A free local scan (no AI, no Groq tokens) measures every file: sharpness, size, look-alikes, colours, faces, and the best moments of each clip.
For each video the agent picks a varied, good-quality set sized to the video (a 5 min review: about 27 photos and 13 clips), avoids footage used in
earlier videos of the same product, and AI-checks only that set. The pick is saved in `selection.json` (the script step makes it, the render step
reuses it); run the script step again for a new pick. `out/footage_selection.md` lists what was picked and why each other file was not.
Force a file in with `assets_pin:` in `facts.yaml`. Settings: `assets:` in `config.yaml` (`select: false` uses everything, as before).
Clips are played from their best moments (sharp and moving) instead of from the start.

## Subscribe / like / share lines
Added automatically (`cta:` and `disclosure: outro*` in `config.yaml`): a **subscribe** ask in the intro (after your first hook line; `intro_position: start` moves it
to the very beginning, `intro: false` turns it off) and **subscribe + like + share** in the outro. Do not write your own; if your script already has a
subscribe line in its opening or closing lines, the agent does not add a second one.

## Groq limits and saved progress
Groq's free tier has a **daily token limit per model**. When it is used up the agent stops at once and tells you (waiting a minute does not help).
Every photo check is saved as it finishes (and the workflow keeps those saved checks for all projects, even after a failed run), so the next run
only checks what is left. To use fewer tokens, photos are shrunk for the AI check (`llm.vision_max_side`), or you can turn the AI photo check off
(`llm.vision_check: false`); then you check your own photos for faces, prices and Amazon page elements.

**Fallback models.** If a model's daily limit is used up, the agent moves to the next one in `llm.vision_fallbacks` / `llm.text_fallbacks`
(config.yaml). Check those names at console.groq.com/docs/models; a name your account does not have is replaced by a model from your own list.

## When a run fails, and storage
- The run's **Summary** page shows the last 60 lines of the log, and the red error box says why it stopped. `run.log` is also in the download.
- If a render fails after the script was written, the script is still saved to the repo: edit `projects/<NAME>/script.md` and run **render**.
- A blocked render shows the exact words that matched the rule and the line.
- GitHub's free plan has only **500 MB** of download storage. Downloads expire after **3 days** on their own. To free space sooner, download your video,
  then run **Actions -> Clean old artifacts** (default: deletes downloads older than 1 day; `0` deletes all). GitHub updates the number every 6 to 12 hours.

## Script safety
Lines that stay unsupported by your facts after a rewrite are **removed** and listed at the bottom of `script.md` ("REMOVED"). The narration never says
"hands-on notes" aloud (a render is blocked if it does). Extra lines are added to their own section, never under the verdict. A key written twice in
`config.yaml` or `facts.yaml` is reported with its line number instead of silently replacing the first.

## Where the affiliate disclosure goes (`disclosure: mode:` in `config.yaml`)
`full` = spoken line + on-screen banner + tag | `on_screen` = silent text near the start + end card | `description_only` = nothing in the video
(shipped setting). Every run writes `description.txt` (Amazon's sentence on its **first line**) and `pinned_comment.txt`. The FTC says a
disclosure inside the video is clearer than one in the description box; `on_screen` is the middle option. You are responsible for compliance.

## Voices
`inworld_*` Inworld (the shipped default; secret `INWORLD_API_KEY`), `edge_*` free (no commercial-licence statement), `eleven_*` ElevenLabs
(free plan: about one video a month, non-commercial, attribution added automatically), `openai_*` pay-as-you-go, `openrouter_voxtral` OpenRouter
(usage-billed). List with `python -m adagent voices`.

**Spending safeguards (all in `config.yaml` -> `voice:`):**
- **Inworld** counts every character in `.adagent_state/tts_usage.json` (saved to the repo by the workflow) and stops at `free_character_budget`
  (40,000, a cautious local cap: check your real allowance in your Inworld account). Going past it needs BOTH `allow_paid_fallback: true` in the
  config AND the workflow box **allow_inworld_paid_tts** for that run.
- **OpenRouter** is billed per use and is blocked unless ALL THREE are set: `allow_paid: true` and `commercial_use_confirmed: true` in the config
  (only after you checked the model's price and licence), and the workflow box **allow_paid_tts** for that run. It is never a backup voice.
- **Backup voice:** only an `edge_*` voice that fails may switch, to the Inworld voice of the same gender. A failing ElevenLabs/OpenAI/Inworld
  voice stops the run instead. Every run writes `out/voice_report.txt` listing which voice read each line and warns if a backup was used.
- Lines already voiced are cached and cost nothing on a re-run.

## Setup
1. Private GitHub repo with this project. 2. Secrets: `GROQ_API_KEY` (AI), `INWORLD_API_KEY` (default voice), optional `ELEVENLABS_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`.
3. Settings -> Actions -> General -> Workflow permissions -> Read and write. 4. `.github/workflows/` holds exactly two files: `make-video.yml` and `clean-artifacts.yml`.

## Guard-rails
Spoken/on-screen rules block prices, price wording, star ratings, review quotes, "best seller", other retailers and health claims. Facts are never
moved between products. Rankings are presented as **your** pick. A winner in a comparison needs 3+ hands_on_notes on every product. Footage is
screened for faces and price/Amazon UI. You still own accuracy, YouTube's rules, music and image rights.

## Test status
Offline (simulated AI/voice): script, comparison and list logic, templates (grade, transitions, 5 caption styles, preview), disclosure modes, scenes,
asset checks, fallback models, paid-voice guards (Inworld budget and ledger, OpenRouter opt-ins, backup voice), full renders under every template. **Not verified:** live Groq / ElevenLabs / Edge / Inworld calls, the real GitHub run, and the AI-clip notebook on a real GPU.

---
name: page-illustrator
description: Generate the single illustration for one page of an illustrated children's book, composed as a left-to-right sequence of moments separated by vertical bars. Use once a page's story text and character conversation are written and an image needs to be generated for it.
---

# Page Illustrator

Read the page's text and conversation (given in the task message), then
generate ONE illustration for the page.

## Composition

Compose the image as a left-to-right sequence of this page's key moments,
matching the order of the conversation, each moment separated from the
next by a clean vertical bar/divider drawn inside the image — like a
filmstrip or comic strip laid out across one wide frame.

The image must include every character mentioned in the conversation.
Characters share the same moment/segment if they interact directly with
each other at that point in the scene.

## Character consistency

The task message includes a CHARACTER REFERENCE: a fixed appearance for
each character established across the book. Render every character
exactly as described there — same skin tone, hair color/style, clothes,
shoes, and accessories — across every moment of this image and across the
whole book. Treat the sequence as one continuous scene: a character's
appearance must not drift, reset, or vary from one moment to the next,
even as their pose, angle, or action changes. Do not redesign or vary any
part of a character's appearance between moments unless the page text
explicitly describes something that changes it partway through (e.g.
changing into pajamas, getting rained on, putting on a costume) — in that
case, keep the new appearance consistent for every moment from that point
onward.

## Style

Japanese anime illustration style: clean bold line art, cel-shaded
coloring with flat color blocks (not painterly/photorealistic), vibrant
saturated palette, soft cinematic lighting. Characters have expressive
faces and large detailed eyes. Backgrounds are lightly stylized (Studio
Ghibli / Makoto Shinkai inspired) rather than photo-real. No text,
watermarks, or speech bubbles in the image.

The speech texts in the conversation will be bubbled onto the image in a
later step — do not draw the bubbles yourself.

## Output

Call `generate_image` exactly once with your prompt, a filename, 
and `all_actions` set to the number of left-to-right moments/actions 
you composed into the image.
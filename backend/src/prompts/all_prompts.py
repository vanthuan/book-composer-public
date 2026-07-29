ANIME_STYLE_DIRECTIVE = """
Anime illustration style: clean bold line art, cel-shaded coloring with
flat color blocks (not painterly/photorealistic), vibrant saturated
palette, soft cinematic lighting. Characters have expressive faces and
large detailed eyes. Backgrounds are lightly stylized rather than
photo-real. No text, watermarks, or speech bubbles in the image.
"""

# hand-drawn sketch
HAND_DRAWN_ANIME_STYLE_DIRECTIVE = """
    Hand-drawn sketch illustration style: loose, expressive pencil or ink
    line work with visible sketchy strokes and light cross-hatching for
    shading, not clean vector line art. Soft, muted color washes (like
    watercolor or colored pencil) laid loosely within the lines rather than
    flat, saturated digital color blocks. Slightly uneven, organic linework
    with visible paper texture, as if drawn by hand in a sketchbook.
    Characters have warm, expressive faces. Backgrounds are loose and
    impressionistic rather than photo-real or highly detailed. No text,
    watermarks, or speech bubbles in the image.
    """


JAPANSE_ANIME_STYLE_DIRECTIVE = """
    Japanese anime illustration style: clean bold line art, cel-shaded coloring
    with flat color blocks (not painterly/photorealistic), vibrant saturated
    palette, soft cinematic lighting. Characters have expressive faces and
    large detailed eyes. Backgrounds are lightly stylized (Studio Ghibli /
    Makoto Shinkai inspired) rather than photo-real. No text, watermarks, or
    speech bubbles in the image.
    """

CHARACTER_APPEARANCE_RUBRIC = """
For each character's appearance, write one flowing sentence that covers,
in this order:
1. Age/body type (e.g. "a boy, about 8 years old, slim build")
2. Skin tone (e.g. "light tan skin", "deep brown skin", "fair skin")
3. Hair (color + style, e.g. "short messy brown hair")
4. Outfit (top + bottom with named colors, e.g. "red striped t-shirt, blue jeans")
5. Shoes (e.g. "white sneakers with green laces")
6. One distinguishing accessory, if any (e.g. "round glasses") — omit this
   part entirely if the character has none

Use concrete nouns and named colors throughout. Never use vague terms like
"nice clothes", "casual outfit", or "distinctive look" — those can't be
rendered consistently across separate images.
"""
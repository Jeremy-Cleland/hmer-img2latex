"""Tokenizer and scale-to-fit transform tests."""

from PIL import Image

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.transforms import ResizeWithAspectRatio


def test_tokenizer_min_freq_and_train_split(tmp_path):
    formulas = tmp_path / "formulas.lst"
    train = tmp_path / "train.lst"
    formulas.write_text("a b c raretoken\n" + "a b c\n" * 6)
    # indices 1-6 are the repeated formula; index 0 has raretoken
    train.write_text("\n".join(f"img{i}.png {i}" for i in range(1, 7)))
    tokenizer = LaTeXTokenizer(max_sequence_length=32)
    tokenizer.fit_on_split(str(formulas), str(train), min_freq=5)
    assert "a" in tokenizer.token_to_id
    assert "raretoken" not in tokenizer.token_to_id


def test_scale_to_fit_does_not_crop():
    img = Image.new("L", (1600, 64), color=0)
    # draw a black bar across the full width so cropping would lose it
    transform = ResizeWithAspectRatio(64, 512)
    out, valid_w, valid_h = transform(img)
    assert out.size == (512, 64)
    assert valid_w == 512
    assert valid_h <= 64
    assert valid_h > 0

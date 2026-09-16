from config import Profile, Region


def test_profile_round_trip(tmp_path):
    path = tmp_path / "profile.json"
    expected = Profile(cast_key="4", template_width_ratio=0.03)
    expected.save(path)
    assert Profile.load(path) == expected


def test_legacy_profile_uses_single_template(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"cast_key":"4","template_file":"old.png","schema_version":1}')
    profile = Profile.load(path)
    assert profile is not None
    assert profile.templates() == ["old.png"]


def test_region_inset_scales_with_resolution():
    assert Region(100, 50, 1000, 800).inset(0.1, 0.2, 0.1) == Region(200, 210, 800, 560)

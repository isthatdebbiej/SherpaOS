from scripts.reindex_hf_collection import assign_group_roles


def test_balanced_roles_cover_every_stratum():
    rows = []
    for cohort in ("high", "low"):
        for category in ("nominal", "mobility", "dynamics", "combined"):
            for group in range(5):
                rows.append(
                    {
                        "cohort_id": cohort,
                        "category": category,
                        "global_scenario_group": f"{cohort}/{category}-{group}",
                        "global_episode_id": f"{cohort}/{category}-{group}",
                    }
                )
    roles = assign_group_roles(rows, 20260830)
    for cohort in ("high", "low"):
        for category in ("nominal", "mobility", "dynamics", "combined"):
            values = [
                role for group, role in roles.items() if group.startswith(f"{cohort}/{category}-")
            ]
            assert values.count("train") == 3
            assert values.count("validation") == 1
            assert values.count("test") == 1

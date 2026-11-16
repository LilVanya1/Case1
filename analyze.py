import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SRC = ROOT / "students.csv"

SNAPSHOT = pd.Timestamp("2026-07-01 00:00:00", tz="Europe/Moscow")
INSTANT_MAX_S = 60
LONG_CHECK_BH = 15
MULTI_RETURN_MIN = 5

EXPECTED_ROLE = {
    "Проверка заявки": "specialist",
    "Согласование руководителем": "manager",
    "Доработка заявки": "requester",
    "Согласование бюджета": "finance",
    "Проверка договора": "legal",
    "Исполнение заявки": "procurement",
    "Внеплановая проверка": "auditor",
}
TYPE_LABEL = {"standard": "обычная", "urgent": "срочная", "contract": "договорная"}


def business_hours(start, end):
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0
    total = 0.0
    day = start.normalize()
    last = end.normalize()
    step = pd.Timedelta(days=1)
    while day <= last:
        if day.weekday() < 5:
            lo = max(start, day + pd.Timedelta(hours=9))
            hi = min(end, day + pd.Timedelta(hours=18))
            if hi > lo:
                total += (hi - lo).total_seconds() / 3600
        day += step
    return total


def load():
    df = pd.read_csv(SRC)
    df["ts"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Europe/Moscow")
    return df.sort_values(["ProcessInstanceId", "EventIndex", "ts"]).reset_index(drop=True)


def build_tasks(df):
    created = df[df.EventType == "task_created"][
        ["ProcessInstanceId", "TaskId", "ActivityName", "EventIndex", "ts"]
    ]
    closed = df[df.EventType == "task_completed"][
        ["TaskId", "ActivityName", "EventIndex", "ts", "UserId", "Role", "Result"]
    ]
    tasks = created.merge(closed, on="TaskId", how="left", suffixes=("_created", "_closed"))
    done = tasks[tasks.ts_closed.notna()].copy()
    done["cal_s"] = (done.ts_closed - done.ts_created).dt.total_seconds()
    done["bh"] = [business_hours(a, b) for a, b in zip(done.ts_created, done.ts_closed)]
    done["off_hours"] = [
        (t.weekday() >= 5) or (t.hour < 9) or (t.hour >= 18) for t in done.ts_closed
    ]
    return done


def route_of(names, results):
    return " > ".join(f"{n}[{r}]" for n, r in zip(names, results))


def build_processes(df, tasks):
    rows = []
    completed = df[df.EventType == "task_completed"]
    for pid, g in df.groupby("ProcessInstanceId", sort=False):
        g = g.sort_values("EventIndex")
        done = g[g.EventType == "task_completed"]
        end = g[g.EventType == "process_completed"]
        start = g.iloc[0]
        rows.append(
            {
                "ProcessInstanceId": pid,
                "RequestType": start.RequestType,
                "Amount": float(start.Amount),
                "StartUserId": start.UserId,
                "StartedAt": start.ts.isoformat(),
                "EndResult": end.Result.iloc[0] if len(end) else "open",
                "EndedAt": end.ts.iloc[0].isoformat() if len(end) else "",
                "NEvents": int(len(g)),
                "NHumanTasks": int(len(done)),
                "Route": route_of(done.ActivityName, done.Result),
                "Open": len(end) == 0,
            }
        )
    procs = pd.DataFrame(rows).set_index("ProcessInstanceId", drop=True)
    max_s = tasks.groupby("ProcessInstanceId").cal_s.max()
    procs["MaxTaskSeconds"] = procs.index.map(max_s)
    returns = (
        completed[
            (completed.ActivityName == "Согласование руководителем")
            & (completed.Result == "return")
        ]
        .groupby("ProcessInstanceId")
        .size()
    )
    procs["ReturnCount"] = procs.index.map(returns).fillna(0).astype(int)
    return procs


def first_index(done, name):
    hit = done[done.ActivityName == name]
    if hit.empty:
        return None
    return int(hit.EventIndex.min())


def detect(df, tasks, procs):
    completed = df[df.EventType == "task_completed"]
    by_proc = {pid: g.sort_values("EventIndex") for pid, g in completed.groupby("ProcessInstanceId")}
    check_med = float(tasks.loc[tasks.ActivityName_created == "Проверка заявки", "bh"].median())
    other_check_max = float(
        tasks.loc[
            (tasks.ActivityName_created == "Проверка заявки") & (tasks.bh <= LONG_CHECK_BH),
            "bh",
        ].max()
    )
    check_p01 = float(tasks.loc[tasks.ActivityName_created == "Проверка заявки", "cal_s"].quantile(0.01))
    rest_max = procs.loc[procs.MaxTaskSeconds >= INSTANT_MAX_S, "MaxTaskSeconds"]
    next_slow = float(rest_max.min()) if len(rest_max) else None
    n_manager = int((completed.ActivityName == "Согласование руководителем").sum())
    n_manager_ok = int(
        (
            (completed.ActivityName == "Согласование руководителем") & (completed.Role == "manager")
        ).sum()
    )
    n_one_return = int((procs.ReturnCount == 1).sum())
    n_mid_return = int(procs.ReturnCount.between(2, MULTI_RETURN_MIN - 1).sum())
    n_with_manager = int(
        completed.loc[completed.ActivityName == "Согласование руководителем", "ProcessInstanceId"].nunique()
    )

    rows = []

    def add(pid, kind, text):
        rows.append({"ProcessInstanceId": pid, "AnomalyType": kind, "Explanation": text})

    for pid, done in by_proc.items():
        meta = procs.loc[pid]
        names = done.ActivityName.tolist()
        label = TYPE_LABEL[meta.RequestType]

        if meta.EndResult == "fulfilled" and "Согласование руководителем" not in names:
            after = "конец журнала"
            if "Проверка заявки" in names:
                i = names.index("Проверка заявки")
                after = names[i + 1] if i + 1 < len(names) else "конец журнала"
            add(
                pid,
                "Пропуск руководителя",
                (
                    f"Заявка {label} на {meta.Amount:.2f} завершилась fulfilled без задания "
                    f"«Согласование руководителем». Следующий этап после «Проверка заявки» — "
                    f"«{after}». Согласование руководителя есть у {n_with_manager} других заявок. "
                    "Норма успешного пути: проверка, руководитель, бюджет, для договорной ещё юрист, "
                    "затем исполнение."
                ),
            )

        bad_role = done[
            (done.ActivityName == "Согласование руководителем") & (done.Role != "manager")
        ]
        if len(bad_role):
            hit = bad_role.iloc[0]
            same = hit.UserId == meta.StartUserId
            who = "это инициатор заявки" if same else "это не инициатор заявки"
            add(
                pid,
                "Чужая роль на согласовании",
                (
                    f"«Согласование руководителем» закрыто с результатом {hit.Result} "
                    f"ролью requester; {who}. На этом этапе роль manager у {n_manager_ok} из "
                    f"{n_manager} закрытий."
                ),
            )

        ie = first_index(done, "Исполнение заявки")
        ib = first_index(done, "Согласование бюджета")
        il = first_index(done, "Проверка договора")
        order_bits = []
        if ie is not None and ib is not None and ie < ib:
            order_bits.append(
                f"«Исполнение заявки» (EventIndex {ie}) раньше «Согласование бюджета» (EventIndex {ib})"
            )
        if ie is not None and il is not None and ie < il:
            order_bits.append(
                f"«Исполнение заявки» (EventIndex {ie}) раньше «Проверка договора» (EventIndex {il})"
            )
        if order_bits:
            add(
                pid,
                "Исполнение раньше бюджета",
                (
                    f"Заявка {label}: " + "; ".join(order_bits) + ". "
                    "Норма: бюджет, для договорной затем юрист, и только после этого закупщик."
                ),
            )

        for i in range(len(names) - 1):
            if names[i] == names[i + 1] == "Проверка заявки":
                a, b = done.iloc[i], done.iloc[i + 1]
                same = a.UserId == b.UserId
                who = "один и тот же пользователь" if same else "разные пользователи"
                add(
                    pid,
                    "Повтор проверки без возврата",
                    (
                        f"Две «Проверка заявки» подряд (EventIndex {int(a.EventIndex)} и "
                        f"{int(b.EventIndex)}), между ними нет «Доработка заявки». Исполнители: {who}. "
                        "Повтор проверки в обычных заявках есть только после return руководителя."
                    ),
                )
                break

        long = tasks[
            (tasks.ProcessInstanceId == pid)
            & (tasks.ActivityName_created == "Проверка заявки")
            & (tasks.bh > LONG_CHECK_BH)
        ]
        if len(long):
            hit = long.sort_values("bh", ascending=False).iloc[0]
            add(
                pid,
                "Сверхдолгая проверка",
                (
                    f"«Проверка заявки» заняла {hit.bh:.1f} рабочих часов "
                    f"({hit.cal_s / 3600:.0f} календарных). Медиана этапа {check_med:.2f} ч, "
                    f"максимум остальных проверок {other_check_max:.1f} ч. "
                    "Между этими значениями других проверок нет."
                    + (
                        " У этой заявки также нет согласования руководителя."
                        if "Согласование руководителем" not in names
                        else ""
                    )
                ),
            )

        if meta.NHumanTasks > 0 and meta.MaxTaskSeconds < INSTANT_MAX_S:
            sub = tasks[tasks.ProcessInstanceId == pid]
            add(
                pid,
                "Мгновенное прохождение",
                (
                    f"Заявка {label} завершилась «{meta.EndResult}». Все {int(meta.NHumanTasks)} "
                    f"человеческих задания закрыты за {sub.cal_s.min():.0f}–{sub.cal_s.max():.0f} с. "
                    f"У остальных заявок самое долгое задание не короче {next_slow:.0f} с. "
                    f"Этап «Проверка заявки» короче остальных, его 1-й перцентиль {check_p01:.0f} с. "
                    f"Порог {INSTANT_MAX_S} с до обычного диапазона этого этапа не доходит."
                ),
            )

        if meta.ReturnCount >= MULTI_RETURN_MIN:
            add(
                pid,
                "Многократный возврат",
                (
                    f"Руководитель вернул заявку {int(meta.ReturnCount)} раз. Каждый return идёт "
                    "в «Доработка заявки» и затем в новую проверку. "
                    f"Один возврат — у {n_one_return} заявок, два–четыре возврата — у {n_mid_return}. "
                    "Переход разрешён схемой; необычна кратность, она не продолжает обычный ряд."
                ),
            )

    out = pd.DataFrame(rows)
    order = [
        "Пропуск руководителя",
        "Чужая роль на согласовании",
        "Исполнение раньше бюджета",
        "Повтор проверки без возврата",
        "Сверхдолгая проверка",
        "Мгновенное прохождение",
        "Многократный возврат",
    ]
    out["AnomalyType"] = pd.Categorical(out.AnomalyType, order, ordered=True)
    return out.sort_values(["AnomalyType", "ProcessInstanceId"]).reset_index(drop=True)


def manual_review(df, procs):
    rows = []
    completed = df[df.EventType == "task_completed"]
    audit_ids = set(
        completed.loc[completed.ActivityName == "Внеплановая проверка", "ProcessInstanceId"]
    )
    for pid in sorted(audit_ids):
        meta = procs.loc[pid]
        n = int((completed.ProcessInstanceId.eq(pid) & completed.ActivityName.eq("Внеплановая проверка")).sum())
        rows.append(
            {
                "ProcessInstanceId": pid,
                "Reason": "unplanned_audit",
                "Detail": (
                    f"{meta.RequestType}, сумма {meta.Amount:.2f}, проверок аудитора {n}, "
                    "результат accept, роль auditor. Редкость сама по себе не нарушение."
                ),
            }
        )
    for pid, meta in procs[procs.Open].iterrows():
        last = df[df.ProcessInstanceId == pid].sort_values("EventIndex").iloc[-1]
        lower_h = (SNAPSHOT - last.ts).total_seconds() / 3600
        rows.append(
            {
                "ProcessInstanceId": pid,
                "Reason": "open_at_snapshot",
                "Detail": (
                    f"Нет process_completed. Последнее событие {last.EventType} "
                    f"«{last.ActivityName}» в {last.ts.isoformat()}. "
                    f"Нижняя граница ожидания до среза {lower_h:.0f} ч. "
                    "В anomalies.csv не включено: открытая заявка на дату выгрузки не аномалия."
                ),
            }
        )
    late_ids = sorted(df.loc[df.ts > SNAPSHOT, "ProcessInstanceId"].unique())
    for pid in late_ids:
        part = df[(df.ProcessInstanceId == pid) & (df.ts > SNAPSHOT)].sort_values("EventIndex")
        rows.append(
            {
                "ProcessInstanceId": pid,
                "Reason": "events_on_snapshot_date",
                "Detail": (
                    f"{len(part)} событий {part.ts.min().isoformat()} — {part.ts.max().isoformat()} "
                    f"({', '.join(part.ActivityName.unique())}). "
                    "Это рабочий день даты среза, не 2 июля и позже. "
                    "В anomalies.csv не включено: маршрут обычный, граница 00:00 двусмысленна."
                ),
            }
        )
    return pd.DataFrame(rows)


def quality_stats(df, tasks, procs):
    human_close = df[df.EventType == "task_completed"]
    role_bad = []
    for name, role in EXPECTED_ROLE.items():
        part = human_close[human_close.ActivityName == name]
        n_bad = int((part.Role != role).sum())
        if n_bad:
            role_bad.append({"activity": name, "expected": role, "n": n_bad})
    variants = procs.Route.value_counts()
    return {
        "n_events": int(len(df)),
        "n_processes": int(procs.shape[0]),
        "fulfilled": int((procs.EndResult == "fulfilled").sum()),
        "rejected": int((procs.EndResult == "rejected").sum()),
        "open": int(procs.Open.sum()),
        "events_after_snapshot": int((df.ts > SNAPSHOT).sum()),
        "processes_after_snapshot": int(df.loc[df.ts > SNAPSHOT, "ProcessInstanceId"].nunique()),
        "index_gaps": int((df.groupby("ProcessInstanceId").EventIndex.diff().dropna() != 1).sum()),
        "time_reversals": int(
            (df.groupby("ProcessInstanceId").ts.diff().dt.total_seconds().dropna() < 0).sum()
        ),
        "off_hours_completions": int(tasks.off_hours.sum()),
        "negative_durations": int((tasks.cal_s < 0).sum()),
        "role_mismatches": role_bad,
        "return_counts": {str(k): int(v) for k, v in procs.ReturnCount.value_counts().sort_index().items()},
        "top_routes": [
            {"route": r, "n": int(n), "share": round(n / len(procs), 4)}
            for r, n in variants.head(8).items()
        ],
        "rework_user_differs_from_initiator": None,
    }


def rework_mismatch(df, procs):
    done = df[df.EventType == "task_completed"]
    rework = done[done.ActivityName == "Доработка заявки"][["ProcessInstanceId", "UserId"]]
    merged = rework.merge(
        procs[["StartUserId"]].reset_index(),
        on="ProcessInstanceId",
    )
    return int((merged.UserId != merged.StartUserId).sum()), int(len(merged))


def write_outputs(df, tasks, procs, anomalies, review, stats):
    DATA.mkdir(exist_ok=True)
    # Без BOM: в шаблоне сдачи заголовок должен совпасть байт в байт.
    anomalies.to_csv(ROOT / "anomalies.csv", index=False, encoding="utf-8")
    task_out = tasks[
        [
            "ProcessInstanceId",
            "TaskId",
            "ActivityName_created",
            "Role",
            "Result",
            "EventIndex_created",
            "EventIndex_closed",
            "ts_created",
            "ts_closed",
            "cal_s",
            "bh",
        ]
    ].rename(
        columns={
            "ActivityName_created": "ActivityName",
            "EventIndex_created": "CreatedEventIndex",
            "EventIndex_closed": "ClosedEventIndex",
        }
    )
    task_out.to_csv(DATA / "tasks.csv", index=False, encoding="utf-8-sig")
    procs.reset_index().to_csv(DATA / "processes.csv", index=False, encoding="utf-8-sig")
    review.to_csv(DATA / "manual_review.csv", index=False, encoding="utf-8-sig")
    variants = (
        procs.Route.value_counts()
        .rename_axis("Route")
        .reset_index(name="NumberOfCases")
    )
    variants.insert(0, "Variant", range(1, len(variants) + 1))
    variants["Share"] = (variants.NumberOfCases / len(procs)).round(4)
    variants.to_csv(DATA / "routes.csv", index=False, encoding="utf-8-sig")
    (
        tasks.groupby("ActivityName_created")
        .agg(
            Tasks=("cal_s", "size"),
            MedianCalendarHours=("cal_s", lambda s: round(s.median() / 3600, 2)),
            MedianBusinessHours=("bh", lambda s: round(float(s.median()), 2)),
            P99BusinessHours=("bh", lambda s: round(s.quantile(0.99), 2)),
            MaxBusinessHours=("bh", lambda s: round(s.max(), 2)),
            MinSeconds=("cal_s", "min"),
        )
        .reset_index()
        .rename(columns={"ActivityName_created": "ActivityName"})
        .to_csv(DATA / "duration_by_activity.csv", index=False, encoding="utf-8-sig")
    )
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def validate(df, procs, anomalies, tasks):
    known = set(df.ProcessInstanceId)
    assert set(anomalies.ProcessInstanceId) <= known
    assert not anomalies.duplicated(["ProcessInstanceId", "AnomalyType"]).any()
    open_ids = set(procs.index[procs.Open])
    assert not set(anomalies.ProcessInstanceId) & open_ids
    audit = set(
        df.loc[df.ActivityName == "Внеплановая проверка", "ProcessInstanceId"]
    )
    # Auditor step alone must not create a row. Overlap with a real type is allowed.
    audit_only_types = set(anomalies.loc[anomalies.ProcessInstanceId.isin(audit), "AnomalyType"])
    assert "Внеплановая проверка" not in set(anomalies.AnomalyType)
    fulfilled_contract = procs.index[(procs.RequestType == "contract") & (procs.EndResult == "fulfilled")]
    legal = set(df.loc[df.ActivityName == "Проверка договора", "ProcessInstanceId"])
    assert set(fulfilled_contract) <= legal
    assert int(tasks.off_hours.sum()) == 0
    counts = anomalies.AnomalyType.value_counts().to_dict()
    expected = {
        "Пропуск руководителя": 16,
        "Чужая роль на согласовании": 8,
        "Исполнение раньше бюджета": 8,
        "Повтор проверки без возврата": 8,
        "Сверхдолгая проверка": 8,
        "Мгновенное прохождение": 8,
        "Многократный возврат": 8,
    }
    assert counts == expected, counts
    long_ids = set(anomalies.loc[anomalies.AnomalyType == "Сверхдолгая проверка", "ProcessInstanceId"])
    skip_ids = set(anomalies.loc[anomalies.AnomalyType == "Пропуск руководителя", "ProcessInstanceId"])
    assert long_ids <= skip_ids
    assert audit_only_types.isdisjoint({"Пропуск руководителя", "Чужая роль на согласовании"})
    return counts


def main():
    df = load()
    tasks = build_tasks(df)
    procs = build_processes(df, tasks)
    anomalies = detect(df, tasks, procs)
    review = manual_review(df, procs)
    differ, rework_n = rework_mismatch(df, procs)
    stats = quality_stats(df, tasks, procs)
    stats["rework_user_differs_from_initiator"] = {"differ": differ, "rework_tasks": rework_n}
    stats["anomaly_rows"] = int(len(anomalies))
    stats["anomaly_processes"] = int(anomalies.ProcessInstanceId.nunique())
    stats["by_type"] = {str(k): int(v) for k, v in anomalies.AnomalyType.value_counts().items()}
    write_outputs(df, tasks, procs, anomalies, review, stats)
    counts = validate(df, procs, anomalies, tasks)
    print("anomalies", len(anomalies), "processes", anomalies.ProcessInstanceId.nunique())
    for k, v in counts.items():
        print(f"  {v:3d}  {k}")
    print("manual", review.Reason.value_counts().to_dict())
    print("rework user differs", differ, "of", rework_n)


if __name__ == "__main__":
    main()

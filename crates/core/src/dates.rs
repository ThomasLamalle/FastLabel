//! UTC timestamps as ISO 8601 strings (`2026-10-06T11:34:09Z`), without a
//! date library.

/// Seconds since the Unix epoch, now.
pub fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
        as i64
}

/// (year, month, day, seconds of the day) of a Unix time.
fn civil(secs: i64) -> (i64, i64, i64, i64) {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d, rem)
}

/// `2026-10-06T11:34:09Z`.
pub fn iso(secs: i64) -> String {
    let (y, m, d, r) = civil(secs);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", r / 3600, r % 3600 / 60, r % 60)
}

/// `20261006-113409`, for file names.
pub fn compact(secs: i64) -> String {
    let (y, m, d, r) = civil(secs);
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", r / 3600, r % 3600 / 60, r % 60)
}

/// Unix time of `YYYY-MM-DDTHH:MM:SS` (anything after is ignored, UTC).
pub fn parse_iso(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'T' | b' ') {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hh, mm, ss) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hh * 3600 + mm * 60 + ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_known_values() {
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(1_791_278_831), "2026-10-06T09:27:11Z");
        assert_eq!(compact(1_791_278_831), "20261006-092711");
        for t in [0, 951_782_400, 1_791_278_831, 4_102_444_799] {
            assert_eq!(parse_iso(&iso(t)), Some(t));
        }
        assert_eq!(parse_iso("2024-02-29 12:00:00.5+02:00"), Some(1_709_208_000));
        assert_eq!(parse_iso("2024-13-01T00:00:00Z"), None);
        assert_eq!(parse_iso("yesterday"), None);
    }
}

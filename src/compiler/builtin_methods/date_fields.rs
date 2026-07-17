//! Date/`Date`-like receiver: zero-arg calls that compile like `GetArrayElement` property access.

/// Method names handled like `d.year` when written as `d.year()` with no arguments.
pub(super) const METHODS: &[&str] = &[
    "year", "month", "quarter", "day", "hour", "minute", "second", "weekday", "to_utc", "utc",
];

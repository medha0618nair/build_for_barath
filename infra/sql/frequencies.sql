-- Athena equivalent of linkage/frequencies.py's compute() (TECHNICAL_SPEC.md §4.5, §6).
--
-- Assumes a Glue table `cases` over the curated parquet the Normalise step
-- writes to S3: one row per canonical case, with each MO_CORE / MO_EXT
-- categorical field as its own string column (mo_core_<field>, mo_ext_<field>)
-- and each TAG_FIELDS field as array<string> (mo_core_tools, mo_core_property_taken).
-- Sentinel tokens (__MISSING__, __ABSENT__, __UNKNOWABLE__) are stored as
-- literal strings in the categorical columns and are simply never members of
-- a tag array.
--
-- These four query shapes are templates: linkage/frequencies.py's compute()
-- runs one per (field, pool) at generation time; writing out all ~21 MO
-- fields literally here would just be repetition of the same shape.

-- 1) same-type, categorical field (u per crime_type, per value)
--    freq = count(this value) / count(any value other than a sentinel)
SELECT
    crime_type,
    mo_core_time_band AS value,
    CAST(COUNT(*) AS DOUBLE) / SUM(COUNT(*)) OVER (PARTITION BY crime_type) AS u
FROM cases
WHERE mo_core_time_band NOT IN ('__MISSING__', '__ABSENT__', '__UNKNOWABLE__')
GROUP BY crime_type, mo_core_time_band;

-- 2) same-type, tag field (independent per-tag rate within a crime_type)
--    denominator is cases where the field was recorded at all (not
--    __MISSING__/__ABSENT__ — an empty-but-recorded tag list still counts)
WITH recorded AS (
    SELECT crime_type, mo_core_tools
    FROM cases
    WHERE mo_core_tools IS NOT NULL   -- __MISSING__/__ABSENT__ mo_core_tools rows are NULL arrays
),
denom AS (
    SELECT crime_type, COUNT(*) AS n FROM recorded GROUP BY crime_type
)
SELECT r.crime_type, tag.value AS tag, CAST(COUNT(*) AS DOUBLE) / d.n AS u
FROM recorded r
CROSS JOIN UNNEST(r.mo_core_tools) AS tag(value)
JOIN denom d ON d.crime_type = r.crime_type
GROUP BY r.crime_type, tag.value, d.n;

-- 3) cross-type, pooled across all ingested property crime (mo_core only —
--    mo_ext isn't in scope for cross-type pairs outside same-family pairs,
--    see linkage/score.py)
SELECT
    mo_core_time_band AS value,
    CAST(COUNT(*) AS DOUBLE) / SUM(COUNT(*)) OVER () AS u
FROM cases
WHERE mo_core_time_band NOT IN ('__MISSING__', '__ABSENT__', '__UNKNOWABLE__')
GROUP BY mo_core_time_band;

-- 4) same-family, mo_ext field (both burglary types pooled — see
--    linkage/schema.py FAMILY and linkage/frequencies.py's same_family pool)
SELECT
    CASE crime_type
        WHEN 'BURGLARY_RESIDENTIAL' THEN 'burglary'
        WHEN 'BURGLARY_COMMERCIAL' THEN 'burglary'
        WHEN 'VEHICLE_THEFT' THEN 'vehicle'
        WHEN 'SNATCHING' THEN 'snatching'
        WHEN 'ATM_TAMPERING' THEN 'atm'
    END AS family,
    mo_ext_entry_point AS value,
    CAST(COUNT(*) AS DOUBLE) / SUM(COUNT(*)) OVER (PARTITION BY
        CASE crime_type
            WHEN 'BURGLARY_RESIDENTIAL' THEN 'burglary'
            WHEN 'BURGLARY_COMMERCIAL' THEN 'burglary'
        END) AS u
FROM cases
WHERE crime_type IN ('BURGLARY_RESIDENTIAL', 'BURGLARY_COMMERCIAL')
  AND mo_ext_entry_point NOT IN ('__MISSING__', '__ABSENT__', '__UNKNOWABLE__')
GROUP BY crime_type, mo_ext_entry_point;

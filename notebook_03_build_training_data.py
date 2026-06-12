# =============================================================================
# CDOCS POC — Notebook 03: Build Training Dataset
# =============================================================================
#
# TARGET LABEL: gfa_name (Governing Functional Area)
#
# When a user uploads a document to CDOCS, the system needs to predict
# which GFA it belongs to. That is what this model learns.
#
# classification and doc_subtype are now FEATURES (document type info),
# not targets. gfa_name is NEVER in the feature text.
#
# INPUT:
#   _dev.edl_app_dev_ops_gsc_ai.datamapping_poc_summaries  (Notebook 02)
#   edf_prd production tables (read-only)
#
# OUTPUT:
#   _dev.edl_app_dev_ops_gsc_ai.datamapping_poc_training
#
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Imports
# -----------------------------------------------------------------------------
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

print("Imports OK")


# -----------------------------------------------------------------------------
# CELL 2 — Configuration
# -----------------------------------------------------------------------------
INPUT_SUMMARIES = "_dev.edl_app_dev_ops_gsc_ai.datamapping_poc_summaries"
OUTPUT_TABLE    = "_dev.edl_app_dev_ops_gsc_ai.datamapping_poc_training"

TBL_DOCUMENTS   = "edf_prd.edf_prd_mmds_ops_cdocs_publish.documents_general"
TBL_ATTACHMENT  = "edf_prd.edf_prd_mmds_ops_cdocs_publish.cdocs_attachment"
TBL_SRC_DETAILS = "edf_prd.edf_prd_mmds_ops_cdocs_publish.documents_source_document_details"
TBL_GFA         = "edf_prd.edf_prd_mmds_ops_cdocs_publish.governing_functional_area_c"
TBL_REGULATION  = "edf_prd.edf_prd_mmds_ops_cdocs_publish.regulation_c"

print("Config OK")


# -----------------------------------------------------------------------------
# CELL 3 — Build metadata view from production tables
# -----------------------------------------------------------------------------
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW cdocs_poc_metadata AS

WITH attachment_one AS (
    SELECT source_id, MIN(attachment_s3_path) AS attachment_s3_path
    FROM {TBL_ATTACHMENT}
    WHERE attachment_s3_path IS NOT NULL
    GROUP BY source_id
),

source_details_one AS (
    SELECT source_id, MIN(object_name) AS object_name
    FROM {TBL_SRC_DETAILS}
    WHERE object_name IS NOT NULL
    GROUP BY source_id
),

base AS (
    SELECT
        d.source_id,
        d.title,
        d.classification,
        d.doc_subtype,
        sd.object_name,
        a.attachment_s3_path,
        gfa.name_v  AS gfa_name,
        reg.name_v  AS regulation_name

    FROM {TBL_DOCUMENTS} d
    LEFT JOIN attachment_one a       ON d.source_id = a.source_id
    LEFT JOIN source_details_one sd  ON d.source_id = sd.source_id
    LEFT JOIN {TBL_GFA} gfa         ON array_contains(d.governing_functional_area__c, gfa.id)
    LEFT JOIN {TBL_REGULATION} reg  ON array_contains(d.regulation__c, reg.id)

    WHERE d.title IS NOT NULL
      AND d.classification IS NOT NULL
)

SELECT
    source_id, title, classification, doc_subtype,
    object_name, attachment_s3_path,
    collect_set(gfa_name)        AS gfa_names,
    collect_set(regulation_name) AS regulation_names
FROM base
GROUP BY source_id, title, classification, doc_subtype,
         object_name, attachment_s3_path
""")

meta_n = spark.sql("SELECT COUNT(*) AS n FROM cdocs_poc_metadata").collect()[0]["n"]
print(f"Metadata view: {meta_n} rows")


# -----------------------------------------------------------------------------
# CELL 4 — Join metadata with LLM summaries → training dataset
#
# TARGET: gfa_name (Governing Functional Area)
#
# FEATURE TEXT includes:
#   doc_subtype    → document type (SOP, policy, report) — now a feature
#   classification → document classification — now a feature
#   summary        → LLM prose summary
#   key_terms      → exact technical terms from the document
#
# FEATURE TEXT excludes:
#   gfa_name          → this IS the target label
#   regulation_names  → strongly correlated with GFA → leakage risk
#   title             → often contains GFA abbreviations → leakage risk
#
# EXPLODE handles multi-GFA documents:
#   A document tagged with 2 GFAs becomes 2 training rows.
#   For POC with ~130 docs this is acceptable.
# -----------------------------------------------------------------------------
training_df = spark.sql(f"""

WITH exploded_meta AS (
    SELECT
        m.source_id,
        m.title,
        m.classification,
        m.doc_subtype,
        m.object_name,
        m.regulation_names,
        gfa_name
    FROM cdocs_poc_metadata m
    LATERAL VIEW EXPLODE(m.gfa_names) t AS gfa_name
    WHERE gfa_name IS NOT NULL
)

SELECT
    em.source_id,
    em.title,
    em.object_name,

    -- TARGET LABEL
    em.gfa_name,

    -- Metadata (for reference, not in features)
    em.classification,
    em.doc_subtype,
    em.regulation_names,

    -- LLM output
    s.filename,
    s.text_length,
    s.quality_score,
    s.llm_confidence,
    s.final_confidence,
    s.routing,
    s.summary,
    s.key_terms,

    -- FEATURE TEXT for ML model
    -- classification and doc_subtype are features (document type signal)
    -- gfa_name is the TARGET — never appears here
    concat_ws(
        ' ',
        coalesce(em.doc_subtype, ''),
        coalesce(em.classification, ''),
        coalesce(s.summary, ''),
        coalesce(array_join(s.key_terms, ' '), '')
    ) AS feature_text,

    -- BASELINE TEXT (no LLM — measures value of LLM step)
    concat_ws(
        ' ',
        coalesce(em.title, ''),
        coalesce(em.object_name, '')
    ) AS baseline_text

FROM exploded_meta em

INNER JOIN {INPUT_SUMMARIES} s
    ON lower(regexp_replace(em.object_name, '\\\\.[^.]+$', ''))
     = lower(regexp_replace(s.filename,    '\\\\.[^.]+$', ''))

WHERE em.gfa_name IS NOT NULL
  AND s.routing = 'ml_training'
  AND s.llm_status = 'success'
  AND length(trim(coalesce(s.summary, ''))) > 0
""")

print(f"Training dataset: {training_df.count()} rows")


# -----------------------------------------------------------------------------
# CELL 5 — Write to Delta table
# -----------------------------------------------------------------------------
(
    training_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Table written : {OUTPUT_TABLE}")
print(f"Row count     : {spark.table(OUTPUT_TABLE).count()}")


# -----------------------------------------------------------------------------
# CELL 6 — Diagnostics
# -----------------------------------------------------------------------------
print("\n=== GFA Label Distribution (TARGET) ===")
spark.sql(f"""
    SELECT gfa_name, COUNT(*) AS n,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM {OUTPUT_TABLE}
    GROUP BY gfa_name
    ORDER BY n DESC
""").display()

print("\n=== Classification Distribution (FEATURE) ===")
spark.sql(f"""
    SELECT classification, COUNT(*) AS n
    FROM {OUTPUT_TABLE}
    GROUP BY classification
    ORDER BY n DESC
""").display()

print("\n=== Feature Text Length ===")
spark.sql(f"""
    SELECT
        CASE
            WHEN length(feature_text) < 200  THEN '< 200'
            WHEN length(feature_text) < 1000 THEN '200-1000'
            WHEN length(feature_text) < 3000 THEN '1000-3000'
            ELSE '> 3000'
        END AS bucket,
        COUNT(*) AS n
    FROM {OUTPUT_TABLE}
    GROUP BY 1 ORDER BY n DESC
""").display()

print("\n=== Join Coverage ===")
spark.sql(f"""
    WITH exploded_meta AS (
        SELECT source_id, object_name, gfa_name
        FROM cdocs_poc_metadata
        LATERAL VIEW EXPLODE(gfa_names) t AS gfa_name
        WHERE gfa_name IS NOT NULL
    )
    SELECT
        CASE WHEN s.filename IS NULL THEN 'no_match'
             ELSE s.routing END AS status,
        COUNT(*) AS n
    FROM exploded_meta em
    LEFT JOIN {INPUT_SUMMARIES} s
        ON lower(regexp_replace(em.object_name, '\\\\.[^.]+$', ''))
         = lower(regexp_replace(s.filename,    '\\\\.[^.]+$', ''))
    GROUP BY 1 ORDER BY n DESC
""").display()


# -----------------------------------------------------------------------------
# CELL 7 — Preview: verify gfa_name is NOT in feature_text
# -----------------------------------------------------------------------------
spark.sql(f"""
    SELECT
        gfa_name,
        classification,
        doc_subtype,
        final_confidence,
        LEFT(summary, 300)       AS summary_preview,
        key_terms,
        LEFT(feature_text, 400)  AS feature_preview
    FROM {OUTPUT_TABLE}
    ORDER BY gfa_name
    LIMIT 20
""").display()


# -----------------------------------------------------------------------------
# CELL 8 — Duplicate check
# If a document appears under multiple GFAs, it creates multiple rows.
# Check how many duplicates exist.
# -----------------------------------------------------------------------------
print("\n=== Documents With Multiple GFAs ===")
spark.sql(f"""
    SELECT source_id, title, collect_set(gfa_name) AS gfa_labels, COUNT(*) AS row_count
    FROM {OUTPUT_TABLE}
    GROUP BY source_id, title
    HAVING COUNT(*) > 1
    ORDER BY row_count DESC
    LIMIT 20
""").display()

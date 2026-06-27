"""Export the AMIGA/CIG optical log(D25) catalogue from the IAA AMIGA database.

This is a provenance utility documenting how ``data/amiga_full_catalogue_logd25.csv``
(already committed to this repository) was produced. It requires access to the
internal IAA AMIGA MySQL server and is not part of the reproducible workflow.

Connection settings are read from environment variables so that no credentials
are ever stored in the source tree:

    AMIGA_DB_HOST   (default: amiga-db.iaa.es)
    AMIGA_DB_USER   (default: amiga)
    AMIGA_DB_PASSWORD   (required)
    AMIGA_DB_PORT   (optional)

Example::

    export AMIGA_DB_PASSWORD=...        # never commit this
    python scripts/export_amiga_optical_diameters.py
"""

import csv
import os

import pymysql


def _connect_with_fallback(host, user, passwd, db_name, port=None):
    attempts = [
        {
            "host": host,
            "user": user,
            "passwd": passwd,
            "db": db_name,
            "cursorclass": pymysql.cursors.DictCursor,
        }
    ]

    if port is not None:
        attempts.append(
            {
                "host": host,
                "user": user,
                "passwd": passwd,
                "port": port,
                "db": db_name,
                "cursorclass": pymysql.cursors.DictCursor,
            }
        )

    last_error = None
    for kwargs in attempts:
        try:
            return pymysql.connect(**kwargs)
        except pymysql.MySQLError as exc:
            last_error = exc

    raise last_error


def fetch_full_catalogue_logd25(host, user, passwd, port=None):
    """
    Return the full AMIGA catalogue with CIG, logd25, E_logd25,
    and PHYS_DISTANCE_decimal.

    This mirrors the "wholeamiga=True" logic by starting from the full
    COORDINATES table and left-joining LEDA and RESULTS_OPT.
    """
    db = _connect_with_fallback(
        host=host,
        user=user,
        passwd=passwd,
        db_name="CIG_RELEASE_2012",
        port=port,
    )

    query = """
        SELECT
            c.cig AS CIG,
            l.logd25 AS logd25,
            l.E_logd25 AS E_logd25,
            r.PHYS_DISTANCE_decimal AS PHYS_DISTANCE_decimal
        FROM CIG_RELEASE_2012.COORDINATES c
        LEFT JOIN CIG_RELEASE_2012.LEDA l
            ON l.cig = c.cig
        LEFT JOIN CIG_RELEASE_2012.RESULTS_OPT r
            ON r.cig = c.cig
        ORDER BY c.cig
    """

    try:
        with db.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        db.close()

    return rows


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "# CIG: AMIGA/CIG galaxy identifier",
                "logd25: logarithm of the optical D25 diameter",
                "E_logd25: uncertainty on logd25",
                "PHYS_DISTANCE_decimal: uncertainty/decimal companion field from RESULTS_OPT",
            ]
        )
        writer.writerow(
            [
                "CIG",
                "logd25",
                "E_logd25",
                "PHYS_DISTANCE_decimal",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["CIG"],
                    row["logd25"],
                    row["E_logd25"],
                    row["PHYS_DISTANCE_decimal"],
                ]
            )


if __name__ == "__main__":
    host = os.environ.get("AMIGA_DB_HOST", "amiga-db.iaa.es")
    user = os.environ.get("AMIGA_DB_USER", "amiga")
    passwd = os.environ.get("AMIGA_DB_PASSWORD")
    port_env = os.environ.get("AMIGA_DB_PORT")
    port = int(port_env) if port_env else None

    if not passwd:
        raise SystemExit(
            "AMIGA_DB_PASSWORD is not set. Export it in your shell before running, e.g.\n"
            "    export AMIGA_DB_PASSWORD=...\n"
            "Credentials must never be hard-coded or committed."
        )

    rows = fetch_full_catalogue_logd25(
        host=host,
        user=user,
        passwd=passwd,
        port=port,
    )

    output_path = os.environ.get(
        "AMIGA_DB_OUTPUT",
        os.path.join("data", "amiga_full_catalogue_logd25.csv"),
    )
    write_csv(rows, output_path)

    print(f"Rows written: {len(rows)}")
    print(f"Saved to: {output_path}")

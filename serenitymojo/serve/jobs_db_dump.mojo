# serenitymojo.serve.jobs_db_dump — read jobs.db back via the pure-Mojo
# sqlite reader and print every row (proves the gallery-index seam).
#   pixi run mojo run -I . -I /home/alex/MOJO-libs serenitymojo/serve/jobs_db_dump.mojo

from sqlite.db import Database


def main() raises:
    var db = Database.open("output/serenity_daemon/jobs.db")
    var rows = db.read_table("jobs")
    print("jobs.db:", len(rows), "finished jobs")
    for i in range(len(rows)):
        var v = rows[i].values.copy()
        print("  id=" + v[0].as_text() + " state=" + v[4].as_text()
              + " model=" + v[2].as_text() + " out=" + v[5].as_text()
              + " created=" + v[1].as_text())
        print("    params_json: " + v[3].as_text())

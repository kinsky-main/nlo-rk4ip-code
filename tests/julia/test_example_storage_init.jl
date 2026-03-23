using Test
using NLOLibExamples

@testset "ExampleRunDB path initialization" begin
    root = mktempdir()
    db = cd(root) do
        ExampleRunDB(joinpath("nested", "db", "example.sqlite3"))
    end

    @test isabspath(db.db_path)
    @test db.db_path == joinpath(root, "nested", "db", "example.sqlite3")
    @test isfile(db.db_path)

    scratch = mktempdir()
    run_group = cd(scratch) do
        begin_group(db, "storage_init_example", nothing)
    end
    @test !isempty(run_group)

    repeated_group = cd(mktempdir()) do
        begin_group(db, "storage_init_example", run_group)
    end
    @test repeated_group == run_group
    @test latest_run_group(db, "storage_init_example") == run_group
end

module job

export Job, Workload, WORKLOADS, job_cluster, job_queue

const WORKLOADS = [
    "ANL-Intrepid-2009-1.swf",
    "CEA-Curie-2011-2.1-cln.swf",
    "CIEMAT-Euler-2008-1.swf",
    "CTC-SP2-1996-3.1-cln.swf",
    "HPC2N-2002-2.2-cln.swf",
    "KIT-FH2-2016-1.swf",
    "LANL-CM5-1994-4.1-cln.swf",
    "LANL-O2K-1999-2.swf",
    "LLNL-Thunder-2007-1.1-cln.swf",
    "LPC-EGEE-2004-1.2-cln.swf",
    "METACENTRUM-2013-3.swf",
    "PIK-IPLEX-2009-1.swf",
    "Sandia-Ross-2001-1.1-cln.swf",
    "SDSC-BLUE-2000-4.2-cln.swf",
    "SDSC-DS-2004-2.1-cln.swf",
    "SDSC-Par-1995-3.1-cln.swf",
    "SDSC-SP2-1998-4.2-cln.swf",
    "SHARCNET-2005-2.swf",
    "SHARCNET-Whale-2005-2.swf"
]

mutable struct Job
    job_id::Int
    cores::Int
    submit_time::Int
    run_time::Int
    requested_time::Int
    simulated_run_time::Int
    simulated_wait_time::Int
end

function Base.show(io::IO, j::Job)
    print(
        io,
        "J", j.job_id,
        "(#", j.cores,
        " W", j.simulated_wait_time,
        " ", j.simulated_run_time,
        "/", j.requested_time,
        ")"
    )
end

Base.:(==)(j1::Job, j2::Job) = j1.job_id == j2.job_id

function Job(words::Vector{SubString{String}})
    job_id = parse(Int, words[1])
    cores = parse(Int, words[5])
    submit_time = parse(Int, words[2])
    run_time = parse(Int, words[4])
    requested_time = parse(Int, words[9])
    if run_time > requested_time
        requested_time = run_time
    end

    Job(job_id, cores, submit_time, run_time, requested_time, 0, 0)
end

function job_cluster(job::Job, max_run_time::Int)
    return fill((job.requested_time - job.simulated_run_time) / max_run_time, job.cores)
end

function job_queue(job::Job, max_run_time::Int, total_cores::Int, available_cores::Int)
    return [job.simulated_wait_time / max_run_time, job.requested_time / max_run_time, job.cores / total_cores, float(available_cores >= job.cores)]
end

struct Workload
    jobs::Vector{Job}
    cores::Int
    max_run_time::Int
end

function Base.show(io::IO, w::Workload)
    print(io, "Workload with ", length(w.jobs), " jobs and ", w.cores, " cores.")
end

function Workload(name)
    jobs = []
    max_run_time = 0
    cores = 0
    println("Loading workload ", name, "...")
    open("data/" * name) do fp
        for line in eachline(fp)
            if line == ""
                continue
            end
            if line[1] == ';'
                if length(line) >= 10 && line[1:10] == "; MaxProcs"
                    cores = parse(Int, split(line, ":")[2])
                end
                continue
            end
            if cores == 0
                error("Improper workload format! Comments must include MaxProcs.")
            end
            j = Job(split(line))
            # filter illegal jobs
            if j.cores > 0 && j.requested_time > 0 && j.run_time > 0
                push!(jobs, j)
                if j.run_time > max_run_time
                    max_run_time = j.run_time
                end
            end
        end
    end
    sort!(jobs, by = j -> j.submit_time)

    Workload(jobs, cores, max_run_time)
end

end

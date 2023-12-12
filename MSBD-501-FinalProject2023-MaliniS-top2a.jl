using MIToS
using MIToS.MSA
using MIToS.Information
using StatsBase
using Plots
using LinearAlgebra
using Distances
using Clustering
using PairwiseListMatrices
using Statistics
using GraphRecipes
using DataFrames
using StatsPlots
using MultivariateStats

# Truncate IJulia outputs at:
ENV["LINES"]   = 20 
ENV["COLUMNS"] = 600;

fasta_file = "top2a.fa"
#fasta_file = ""
msa = read(fasta_file, FASTA, generatemapping=true, useidcoordinates=true)
println("This MSA has ", nsequences(msa), " sequences...")
#fig = plotmsa(msa; colorscheme = :tableau_blue_green)

msa = msa[:,556:1000]

coverage(msa)

columngapfraction(msa)

gr(size=(600,300))

println("\tsequences\tcolumns")
println( "Before:\t", nsequences(msa), "\t\t", ncolumns(msa)  )
# delete sequences with less than 90% coverage of the MSA length:
filtersequences!(msa, coverage(msa) .>= 0.9)
# delete columns with more than 10% of gaps:
filtercolumns!(msa, columngapfraction(msa) .<= 0.1)
println( "After:\t", nsequences(msa), "\t\t",  ncolumns(msa)  )

histogram(  vec(columngapfraction(msa)),
            # Using vec() to get a Vector{Float64} with the fraction of gaps of each column
            xlabel = "gap fraction in [0,1]", bins = 10, legend = false)

histogram(  vec(coverage(msa) .* 100.0), #  Column with the coverage of each sequence
            xlabel = "coverage [%]", legend=false)

pid = percentidentity(msa)

pidtable = to_table(pid, diagonal=false)

quantile(convert(Vector{Float64}, pidtable[:,3]), [0.00, 0.25, 0.50, 0.75, 1.00])

meanpercentidentity(msa)

gr()
heatmap(convert(Matrix, pid), yflip=true, ratio=:equal)

histogram(pidtable[:,3], xlabel ="Percentage of identity", legend=false)

getresidues(msa)

getresiduesequences(msa)

sequencenames(msa)

msa1 = MIToS.MSA.gapstrip(msa,gaplimit=0.65)

residues = getresidues(msa1) # estimateincolumns functions take a Matrix{Residue}

Hx = mapcolfreq!(entropy, msa, Counts(ContingencyTable(Float64, Val{1}, UngappedAlphabet())))

#Hxy = mapcolpairfreq!(entropy, msa1, Counts(ContingencyTable(Float64, Val{2}, UngappedAlphabet())))

Time_Pab = map(1:100) do x
    time = @elapsed mapcolpairfreq!(entropy, msa, Probabilities(ContingencyTable(Float64, Val{2}, UngappedAlphabet())))
end

Time_Nab = map(1:100) do x
    time = @elapsed mapcolpairfreq!(entropy, msa, Counts(ContingencyTable(Float64, Val{2}, UngappedAlphabet())))
end

gr()

histogram( [Time_Pab Time_Nab],
    labels = ["Using ResidueProbability" "Using ResidueCount"],
    xlabel = "Execution time [seconds]" )

NMIxy = mapcolpairfreq!(normalized_mutual_information, msa1, Counts(ContingencyTable(Float64, Val{2}, GappedAlphabet())), Val{false})

NMIxy_transpose = mapcolpairfreq!(normalized_mutual_information, transpose(msa1), Counts(ContingencyTable(Float64, Val{2}, GappedAlphabet())), Val{false})

NMI_matrix_t = convert(Matrix{Float64}, NMIxy_transpose.array)

num_cluster = 5

# cluster X into 20 clusters using K-means 
KClusters = kmeans(NMI_matrix_t, num_cluster; maxiter = 100,  
                                display=:iter)

# verify the number of clusters 
nclusters(KClusters) == 5

# get the assignments of points to clusters 
cluster_assignment = assignments(KClusters) 

# get the cluster sizes 
cnt = counts(KClusters) 

# get the cluster centers 
Cluster_Center = KClusters.centers 

df = DataFrame( seqnum = 1:nsequences(msa1),
                seqname = sequencenames(msa1),
                cluster = cluster_assignment, # the cluster number/index of each sequence
                coverage = vec(coverage(msa1)))

first(df, 20)

top_clusters = combine(
    groupby(df, :cluster),
    :seqnum => length => :n_sequences,
    :coverage => mean => :mean_coverage
)

# Assuming you already have the df DataFrame with the specified columns
df_filtered = filter(row -> row.cluster > 1, df)

first(df_filtered, 20)

h = @df df histogram(:cluster, ylabel="nseq")

maxcoverage = by(df, :cluster, cl -> cl[ findmax(cl[:coverage])[2] ,
                 [:seqnum, :seqname, :coverage]])

first(maxcoverage, 20)

h = @df maxcoverage histogram(:cluster, ylabel="nseq")

cluster_references = Bool[ seqnum in maxcoverage[:seqnum] for seqnum in 1:nsequences(msa1) ]

filtersequences!(msa1, cluster_references)

sum(KClusters.assignments.==5)

graphplot(NMI_matrix_t[1:10,1:10])

graphplot(UpperTriangular(NMI_matrix_t[1:10,1:10]))

scatter(KClusters.assignments)

heatmap(log2.(NMI_matrix_t), yflip=true)

# Create a scatter plot
scatter(1:size(NMI_matrix_t, 2), NMI_matrix_t, color=cluster_assignment, legend=false)
xlabel!("position")
ylabel!("seq")

# Perform PCA
pca_result = fit(PCA, NMI_matrix_t', maxoutdim=2)
pca_data = MultivariateStats.transform(pca_result, NMI_matrix_t')

# Create a PCA plot
scatter(pca_data[1, :], pca_data[2, :], color=cluster_assignment, legend=false)
xlabel!("Principal Component 1")
ylabel!("Principal Component 2")

scatter3d(NMI_matrix_t[1, :], NMI_matrix_t[2, :], NMI_matrix_t[3, :], color=cluster_assignment, legend=false)
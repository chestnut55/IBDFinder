setwd("C:/Users/0/Desktop/patent/IBD/Experients/spieceasi")
library(devtools)
install_github("zdk123/SpiecEasi")
library(SpiecEasi)
library(Matrix)

ibd_otus <- read.csv('input/ibd_otus_num_1.csv', header = TRUE)
ibd_otus <- data.matrix(ibd_otus)
depths <- rowSums(ibd_otus)
ibd_otus.n  <- t(apply(ibd_otus, 1, norm_to_total))
ibd_otus.cs <- round(ibd_otus.n * min(depths))

d <- ncol(ibd_otus.cs)
n <- nrow(ibd_otus.cs)
e <- d

set.seed(10010)
graph <- make_graph('cluster', d, e)
Prec  <- graph2prec(graph)
Cor   <- cov2cor(prec2cov(Prec))

X <- synth_comm_from_counts(ibd_otus.cs, mar=2, distr='zinegbin', Sigma=Cor, n=n)

se <- spiec.easi(X, method='mb', lambda.min.ratio=1e-2, nlambda=15)
# Applying data transformations...
# Selecting model with pulsar using stars...
# Fitting final estimate with mb...
# done
huge::huge.roc(se$est$path, graph, verbose=FALSE)
stars.pr(getOptMerge(se), graph, verbose=FALSE)
# stars selected final network under: se.est$refit$stars

se.mb.amgut <- spiec.easi(ibd_otus, method='mb', lambda.min.ratio=1e-2,
                          nlambda=20, pulsar.params=list(rep.num=50))
se.gl.amgut <- spiec.easi(ibd_otus, method='glasso', lambda.min.ratio=1e-2,
                          nlambda=20, pulsar.params=list(rep.num=50))
sparcc.amgut <- sparcc(ibd_otus)
## Define arbitrary threshold for SparCC correlation matrix for the graph
sparcc.graph <- abs(sparcc.amgut$Cor) >= 0.3
diag(sparcc.graph) <- 0
sparcc.graph <- Matrix(sparcc.graph, sparse=TRUE)
## Create igraph objects
ig.mb     <- adj2igraph(getRefit(se.mb.amgut))
ig.gl     <- adj2igraph(getRefit(se.gl.amgut))
ig.sparcc <- adj2igraph(sparcc.graph)

## set size of vertex proportional to clr-mean
vsize    <- rowMeans(clr(ibd_otus, 1))+6
am.coord <- layout.fruchterman.reingold(ig.mb)

par(mfrow=c(1,3))
plot(ig.mb, layout=am.coord, vertex.size=vsize, vertex.label=NA, main="MB")
plot(ig.gl, layout=am.coord, vertex.size=vsize, vertex.label=NA, main="glasso")
plot(ig.sparcc, layout=am.coord, vertex.size=vsize, vertex.label=NA, main="sparcc")

dd.gl     <- degree.distribution(ig.gl)
dd.mb     <- degree.distribution(ig.mb)
dd.sparcc <- degree.distribution(ig.sparcc)

plot(0:(length(dd.sparcc)-1), dd.sparcc, ylim=c(0,.35), type='b',
     ylab="Frequency", xlab="Degree", main="Degree Distributions")
points(0:(length(dd.gl)-1), dd.gl, col="red" , type='b')
points(0:(length(dd.mb)-1), dd.mb, col="forestgreen", type='b')
legend("topright", c("MB", "glasso", "sparcc"),
       col=c("forestgreen", "red", "black"), pch=1, lty=1)


pargs <- list(seed=10010)
se <- spiec.easi(ibd_otus, method='mb', lambda.min.ratio=5e-1, nlambda=10, pulsar.params=pargs)
# Warning in pulsar(data = X, fun = match.fun(estFun), fargs = args, seed =
# 10010, : Optimal lambda may be smaller than the supplied values
getOptInd(se)
sum(getRefit(se))/2

se <- spiec.easi(ibd_otus, method='mb', lambda.min.ratio=1e-1, nlambda=10, pulsar.params=pargs)
getStability(se)
sum(getRefit(se))/2

se <- spiec.easi(ibd_otus, method='mb', lambda.min.ratio=1e-1, nlambda=100, pulsar.params=pargs)
getStability(se)
sum(getRefit(se))/2



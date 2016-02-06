# nn.R     J. Rohwer   Oct 2015
# simple feed-forward multilayer neural network
# trainable by stochastic gradient descent with momentum
# todo: add dropout and convnet options

# Some activation functions to try
activation_funcs <- list(
    logi=function(z) {1/(1+exp(-z))},
    tanh=function(z) {0.5*tanh(pi*z) + 0.5},
    sin=sin,
    relu=function(z) {z[z<0] <- 0; z})

# 1st derivatives of activation functions
d_activation_funcs <- list(
    logi=function(z) {z <- 1/(1+exp(-z)); z*(1-z)},
    tanh=function(z) {0.5*pi / cosh(pi*z)^2},
    sin=cos,
    relu=function(z) {z[z<0] <- 0; z[z>0] <- 1; z})

# 1st deriv of some cost functions
dcostf <- list(
    quad=function(y, a) {a-y},              # quadratic
    xe=function(y, a) {(a-y) / (a*(1-a))})  # cross-entropy

# Create a simple feedforward neural network
network <- function(spec, batchsz, eta, mo, cost) {
    # This is here just to translate something like '784/tanh 30/tanh 10' to a
    # matrix of input size, output size, and activation func for each layer
    trspec <- function(spec) {
        mvec <- function(re, txt) {unlist(regmatches(txt, gregexpr(re, txt)))}
        lrtxt <- unlist(strsplit(spec, '/'))
        lrsz <- mvec('[0-9]+', lrtxt)
        actv <- mvec('[a-z]+', lrtxt)
        cbind(lrsz[-length(lrsz)], lrsz[-1], actv)
    }
    # Given character vector of input, output, activation func create a layer for the net
    # with random normally distributed weights and biases.
    layer <- function(a) {
        v <- as.numeric(a[-length(a)])
        list(W=matrix(rnorm(prod(v), sd=0.1), ncol=v[1]),
             B=matrix(rnorm(v[2], sd=0.1), ncol=1),
             actv=activation_funcs[[a[length(a)]]],
             dactv=d_activation_funcs[[a[length(a)]]])
    }
    net <- list(lrs=apply(trspec(spec), 1, layer))
    params <- list(batchsz=batchsz, eta=eta, mo=mo, dcost=dcostf[[cost]])
    net[names(params)] <- params
    net
}

# Some helpers to abbreviate indexing parts of the net.
# todo: learn about R objects and try to improve this.
A_prev <- function(net, i, X) {if (i==1) X else net$lrs[[i-1]]$A}  # activation of previous layr
n_prev <- function(net, i) {dim(net$lrs[[i]]$W)[2]}                # size of previous layer
sz <- function(net) {length(net$lrs)}                   # number of layers in the net
W <- function(net, i) {net$lrs[[i]]$W}                  # weights at this layer
B <- function(net, i) {net$lrs[[i]]$B}                  # biases at this layer
Z <- function(net, i) {net$lrs[[i]]$Z}                  # input at this layer
A <- function(net, i) {net$lrs[[i]]$A}                  # activation at this layer
delta <- function(net, i) {net$lrs[[i]]$delta}       # delta at this layer
actv <- function(net, i) {net$lrs[[i]]$actv}         # activation func for this layer
dactv <- function(net, i) {net$lrs[[i]]$dactv}       # deriv of activation func for this layer
dW_prev <- function(net, i) {net$lrs[[i]]$dW_prev}   # previous step weight update (for momentum)
dB_prev <- function(net, i) {net$lrs[[i]]$dB_prev}   # previous step bias update (for momentum)

# Propagate a batch of examples through the network
# net: net object with input size matching X
# X: is a matrix of input activations, one column per example
# Value: a net object with activations from applying X
forward <- function(net, X) {
    for (i in 1:sz(net)) {
        net$lrs[[i]]$Z <- W(net, i) %*% A_prev(net, i, X) / n_prev(net, i) + rep(B(net, i), dim(X)[2])
        net$lrs[[i]]$A <- actv(net, i)(Z(net, i))
    }
    net
}

# Backpropagate error through each layer of net
# net: net object with input size matching X *and activations already corresponding to X*
# X: batch of input examples
# Y: corresponding batch of target outputs
# Value: a net object with average cost gradient for the batch at each layer ("delta")
backprop <- function(net, X, Y) {
    for (i in (n<-sz(net)):1) {
        dcost <- if (i==n) net$dcost(Y, A(net, n)) else t(W(net, i+1)) %*% delta(net, i+1)
        net$lrs[[i]]$delta <- dcost * dactv(net, i)(net$lrs[[i]]$Z)
    }
    net
}

# Stochastic gradient descent (one iteration)
# net: net object
# eta: learning rate (fraction of gradient by which to update weights)
# X: input batch
# Y: target outputs
# Value: a net object with weights and biases updated by eta*gradient for the batch
sgd <- function(net, eta, X, Y) {
    for (i in sz(net):1) {
        net$lrs[[i]]$W <- W(net, i) - eta*delta(net, i) %*% t(A_prev(net, i, X))
        net$lrs[[i]]$B <- B(net, i) - eta*sum(delta(net, i))
    }
    net
}

# Stochastic gradient descent with momentum (one iteration)
# net: net object
# eta: learning rate (fraction of gradient by which to update weights)
# mo: momemtum (like 1st order IIR, fraction of previous update vector to keep)
# X: input batch
# Y: target outputs
# Value: a net object with weights and biases updated by eta*gradient for the batch
#        and with momentum (1st order) tracked by "dW_prev" and "dB_prev"
sgdmo <- function(net, eta, mo, X, Y) {
    null_to_0 <- function(x) {if (is.null(x)) 0 else x}
    for (i in sz(net):1) {
        dW <- -eta*(1-mo)*delta(net, i) %*% t(A_prev(net, i, X)) + mo*null_to_0(dW_prev(net, i))
        dB <- -eta*(1-mo)*sum(delta(net, i)) + mo*null_to_0(dB_prev(net, i))
        net$lrs[[i]]$W <- W(net, i) + dW
        net$lrs[[i]]$B <- B(net, i) + dB
        net$lrs[[i]]$dW_prev <- dW
        net$lrs[[i]]$dB_prev <- dB
    }
    net
}

# Train the network on the given examples via SGD for the specified number of epochs.
# Params taken from the net object:
#   batchsz: num training samples per batch, averaged to obtain gradient at each step.
#   eta: learning rate as a function of epoch number
#   mo: momentum as a function of epoch number
# net: net object
# data: list with X (input) and Y (target output) matrices, one example per column
# epochs: numeric vector of epoch indices to train
#         provided this way to functions that adapt hyper-parameters to epoch number
#         will work correctly when resuming training
# report_batch: function to call with details of results of each batch, or NULL
# report_epoch: function to call after each epoch, or NULL
# Value: net object with final weights and biases after training
train <- function(net, data, epochs, report_batch=NULL, report_epoch=NULL) {
    step <- function(net, eta, mo, X, Y) {
        net <- forward(net, X)
        net <- backprop(net, X, Y)
        sgdmo(net, eta, mo, X, Y)
    }
    for (i in epochs) {
        batches <- matrix(sample.int(dim(data$X)[2]), nrow=net$batchsz)
        epochsz <- dim(batches)[2]
        eta <- net$eta(i)/net$batchsz
        mo <- net$mo(i)
        for (j in 1:epochsz) {
            k <- batches[, j]            # indices of examples for this batch
            X <- as.matrix(data$X[, k])  # inputs for this batch
            Y <- as.matrix(data$Y[, k])  # target outputs
            net <- step(net, eta, mo, X, Y)
            if (!is.null(report_batch)) report_batch(net, X, Y)
        }
        if (!is.null(report_epoch)) report_epoch(net, i)
    }
    net
}

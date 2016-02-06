# mnist.R     J. Rohwer   Oct 2015
# learning R by training a digit classifier on the MNIST dataset

# Obtain MNIST dataset from: http://yann.lecun.com/exdb/mnist/
fname_train_data   <- 'data/train-images-idx3-ubyte'
fname_train_labels <- 'data/train-labels-idx1-ubyte'
fname_test_data    <- 'data/t10k-images-idx3-ubyte'
fname_test_labels  <- 'data/t10k-labels-idx1-ubyte'

# Load the MNIST dataset (format described at http://yann.lecun.com/exdb/mnist/)
# Value: list containing train and test datasets
# Each dataset has matrices X and Y of inputs and outputs (one example per column)
# As well as a matrix "digits" containing the output labels (as opposed to the one-hot encoding of Y)
load_mnist <- function() {
    read_int32 <- function(f) {readBin(f, 'integer', n=1, size=4, endian='big')}
    load_data <- function(fname_images, fname_labels) {
        f    <- file(fname_images, 'rb')
        code <- read_int32(f)
        n    <- read_int32(f)
        nrow <- read_int32(f)
        ncol <- read_int32(f)
        X    <- matrix(readBin(f, 'integer', n=n*nrow*ncol, size=1, signed=FALSE), ncol=n)/255
        close(f)
        f    <- file(fname_labels, 'rb')
        code <- read_int32(f)
        n    <- read_int32(f)
        digits <- matrix(readBin(f, 'integer', n=n, size=1, signed=FALSE), ncol=n)
        close(f)
        Y    <- apply(digits, 2, function(a) {v <- rep(0, 10); v[a+1] <- 1; v})
        list(X=X, Y=Y, digits=digits)
    }
    list(train=load_data(fname_train_data, fname_train_labels),
         test=load_data(fname_test_data, fname_test_labels))
}

# plot a single example and the given index
show_digit <- function(data, i) {
    image(matrix(data$X[ ,i], ncol=28)[, 28:1], col=gray(0:255/255))
    title(data$digits[, i])
}

# plot images of the digits at all the provided indices each in its own subplot
show_digits <- function(data, i) {
    layout(matrix(1:length(i), ncol=floor(sqrt(length(i)))))
    sapply(i, function(i) {show_digit(data, i)})
}

# join indexed input samples into single image and plot
mnist_image <- function(X, m, s) {
    m <- t(m)
    nrow <- dim(m)[1]
    ncol <- dim(m)[2]
    res <- matrix(0, nrow=nrow*28, ncol=ncol*28)
    for (y in 1:nrow) {
        for (x in 1:ncol) {
            res[1:28 + (y-1)*28, 1:28 + (x-1)*28] <- X[ ,m[y,x]]
        }
    }
    image(res[, dim(res)[2]:1], col=gray(0:255/255))
    title(s)
}

# create an mnist data object containing just the indexed samples, useful for faster testing
data_subset <- function(data, i) {
    list(X=data$X[, i], Y=data$Y[, i], digits=matrix(data$digits[, i], nrow=1))
}

# forward propagate X and return the fraction of predictions matching Y
# Value: classification accuracy (fraction correct)
test_mnist <- function(net, X, Y) {
    out <- A(forward(net, X), sz(net))
    correct <- apply(out, 2, which.max) == apply(Y, 2, which.max)
    sum(correct) / length(correct)
}

# forward propagate X and return indices of misclassified inputs (those not matching Y)
mnist_misclassified <- function(net, X, Y) {
    out <- A(forward(net, X), sz(net))
    (1:dim(X)[2])[apply(out, 2, which.max) != apply(Y, 2, which.max)]
}

# a rough way to see which examples the classifier succeeds with "easily"
# take the top n correct responses ordered by difference in max and mean of output
# (would also be interesting to see high confidence mis-classifications)
mnist_well_classified <- function(net, X, Y, n) {
    out <- A(forward(net, X), sz(net))
    out <- out[, apply(out, 2, which.max) == apply(Y, 2, which.max)]        # just take correct responses
    order(apply(out, 2, max) - apply(out, 2, mean), decreasing=TRUE)[1:n]   # take top n by this measure of confidence
}

# get the n least confident classifications whether or not they are correct
# note: consider other confidence measures
mnist_least_confident <- function(net, X, Y, n) {
    out <- A(forward(net, X), sz(net))
    order(apply(out, 2, max) - apply(out, 2, mean))[1:n]
}

# ----------
# Try it out
#

# lazy load mnist data (expected at data paths at top of file)
if (!exists('mnist')) {
    cat('Loading MNIST\n')
    mnist <- load_mnist()
}

# expects th neural network code in working directory
source('nn.r')

# train a network (and test on entire test set after each epoch, and print progress)
net <- network('784/tanh 30/tanh 10', b=10, eta=function(e){0.02*0.95^e}, mo=function(e){0.95-0.9^e}, co='xe')
cat('Training on MNIST\n')
net <- train(net, mnist$train, 1:10, report_epoch=function(net, i) {
    cat(sprintf("epoch %d: %.2f%%\n", i, test_mnist(net, mnist$test$X, mnist$test$Y)*100))
    flush.console()
})

# write some images showing what we got right and wrong
save_image_10x10 <- function(i, desc) {
    png(paste(desc, ".png", sep=""))
    mnist_image(mnist$test$X, matrix(i, ncol=10), desc)
    dev.off()
}
save_image_10x10(mnist_misclassified(net, mnist$test$X, mnist$test$Y)[1:100], "misclassified")
save_image_10x10(mnist_well_classified(net, mnist$test$X, mnist$test$Y, 100), "well_classified")
save_image_10x10(mnist_least_confident(net, mnist$test$X, mnist$test$Y, 100), "low_confidence")

This is a image caption model independent of the outer project, but would later be used to calculate a consistency loss.

If use cross-entropy loss, the loss can be back-propagated, but this kinds of loss is too rigid.

If use a GAN loss, then how to back-propagate the loss is a question, because the words are generated discretely in a sequential manner.
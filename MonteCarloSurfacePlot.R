library(plotly)
valueFunction <- read.csv("valueFunction.csv")


mat <- matrix(nrow = 21,ncol=10)

for (i in 1:nrow(valueFunction)){
  
  mat[valueFunction[i,]$player,valueFunction[i,]$dealer] <- valueFunction[i,]$value
}

plot_ly(z=mat,type="surface") %>% layout(
  title = "MonteCarlo Optimal Value Function After 2000000 episodes",
  scene = list(
    xaxis = list(title = "Dealer's Hand"),
    yaxis = list(title = "Player's Hand"),
    zaxis = list(title = "Value Function")
  ))

htmlwidgets::saveWidget(as_widget(plot_ly(z=mat,type="surface") %>% layout(
  title = "MonteCarlo Optimal Value Function After 2000000 episodes",
  scene = list(
    xaxis = list(title = "Dealer's Hand"),
    yaxis = list(title = "Player's Hand"),
    zaxis = list(title = "Value Function")
  ))), "surfacePlot.html")

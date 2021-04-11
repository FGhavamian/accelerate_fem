import React from "react";
import Plotly from "plotly.js";
import createPlotlyComponent from "react-plotly.js/factory";
const Plot = createPlotlyComponent(Plotly);

function Histogram({ name, data }) {
  const dataProcessed = data.map((d) => ({
    type: "histogram",
    x: d["data"],
    nbinsx: 100,
    name: d["name"],
    opacity: 0.3,
    histnorm: 'probability',
  }));

  return (
    <Plot
      data={dataProcessed}
      layout={{
        width: 1080,
        height: 640,
        barmode: "overlay",
        title: name,
        xaxis: { title: "Max plastic strain x 1e4" },
        yaxis: { title: "Count" },
      }}
    />
  );
}

export default Histogram;

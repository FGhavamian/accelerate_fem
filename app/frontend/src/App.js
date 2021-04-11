import { useState } from "react";
import "./App.css";
import computeResults from "./apis/ml";
import Histogram from "./components/histogram";

const name = "Max plastic strain";

const cases = [
  {
    name: "20-25",
    n_samples: 1000,
    radius_range: [20, 25],
  },
  {
    name: "45-50",
    n_samples: 1000,
    radius_range: [45, 50],
  },
  {
    name: "20-50",
    n_samples: 1000,
    radius_range: [20, 50],
  },
];

function App() {
  const [results, setResults] = useState([]);

  return (
    <div className="app">
      <h1 onClick={() => computeResults(name, cases, setResults)}>Hi</h1>
      <Histogram name={name} data={results} />
    </div>
  );
}

export default App;

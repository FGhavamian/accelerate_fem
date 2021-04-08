import axios from "axios";

const computeResults = async (name, cases, setResults) => {
  const results = await axios.put("http://127.0.0.1:8000/predict", cases, {
    params: {
      name: name,
    },
  });

  setResults(results.data);
};

export default computeResults;

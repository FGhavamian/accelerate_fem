import axios from "axios";

const computeResults = async (cases, setResults) => {
  const results = await axios.put("http://127.0.0.1:8000/predict", cases, {
    params: {
      name: "test3",
    },
  });

  setResults(results.data);
};

export default computeResults;

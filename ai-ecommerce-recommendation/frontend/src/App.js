import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [product, setProduct] = useState("");
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);

// const fetchRecommendations = () => {
//   if (!product) {
//     alert("Enter product");
//     return;
//   }

//   axios.get(`http://127.0.0.1:5000/recommend?product=${product}`)
//     .then(res => {
//       console.log("API Response:", res.data);  // 👈 DEBUG
//       setProducts(res.data);
//     })
//     .catch(err => console.error(err));
// };
const fetchRecommendations = () => {
  if (!product) {
    alert("Enter product");
    return;
  }

  setLoading(true);

  // axios.get(`http://127.0.0.1:5000/recommend?product=${product}`)
  //   .then(res => {
  //     setProducts(res.data);
  //     setLoading(false);
  //   })
  axios.get(`http://127.0.0.1:5000/recommend?product=${product}`)
  .then(res => {
    console.log("API Response:", res.data);   // ✅ CORRECT PLACE
    setProducts(res.data);
  })
    .catch(err => {
      console.error(err);
      setLoading(false);
    });
};
  return (
    <div className="container">
      <h1>🛒 AI Product Recommendations</h1>

      <input
        type="text"
        placeholder="Enter product (e.g. shoes)"
        value={product}
        onChange={(e) => setProduct(e.target.value)}
      />

      <button onClick={fetchRecommendations}>Search</button>

      {/* <div className="grid">
        {products.map((p, i) => (
          <div className="card" key={i}>
            <img src={`https://source.unsplash.com/200x200/?${p}`} alt={p} />
            <h3>{p}</h3>
            <p>₹{Math.floor(Math.random() * 5000) + 500}</p>
  <button>Add to Cart</button>
          </div>
        ))}
      </div> */}
      {loading ? (
  <p>Loading...</p>
) : (
  <div className="grid">
    {products.map((p, i) => (
      <div className="card" key={i}>
        <img src={`https://source.unsplash.com/200x200/?${p}`} alt={p} />
        <h3>{p}</h3>
        <p>₹{Math.floor(Math.random() * 5000) + 500}</p>
        <button>Add to Cart</button>
      </div>
    ))}
  </div>
)}
    </div>
  );
}

export default App;
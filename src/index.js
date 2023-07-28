var connect = require("connect");
var serveStatic = require("serve-static");
var fs = require('fs');
var express = require('express')
const app = express()
const cors = require('cors')
const corsOptions = {
  origin:'*',
  mode: 'no-cors',
  credentials:true,
  optionSuccessStatus:200
}

app.use(cors(corsOptions))

// app.get('/',(req,res) => {
//   // res.sendFile(__filename,'index.html')
//   res.json({ msg: 'helloooooo'})
// })


// app.listen(8080, () => {
//   console.log(`Server is running on http://localhost:8080`);
// });

// connect()
//   .use(serveStatic(__dirname))
//   .listen(8080, function() {
//     console.log("Server running on 8080...");
//   });

  app
  .use(serveStatic(__dirname))
  .listen(8080, function() {
    console.log("Server running on 8080...");
  });

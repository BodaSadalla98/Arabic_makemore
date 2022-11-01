const toEnName = require("arabic-name-to-en");

console.log(toEnName('الجوهرة'))

const readline = require('readline');
const fs = require('fs');

const rl = readline.createInterface({
  input: fs.createReadStream('combined_arabic_names.txt')
});

rl.on('line', (line) => {
//   console.log('Line from file:', line);
  fs.appendFile('combined_english_names.txt', toEnName(line) + '\n',function (err) {
    if (err) return console.log(err);
    console.log('Appended!');
 });
});
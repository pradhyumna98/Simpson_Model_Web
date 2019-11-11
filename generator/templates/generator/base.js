// var x = document.getElementById('sel1')
x='',i;
for (i=1; i<=6; i++) {
  x = x + "<h" + i + ">Heading " + i + "</h" + i + ">";
}
document.getElementById("demo").innerHTML = x;
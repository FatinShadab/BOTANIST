let button = document.getElementById("potato_leave_image_submit")
let input = document.getElementById("potato_leave_image")

input.addEventListener("input", function(e) {
	if(input.value.length == 0) {
  	button.disabled = true
  } else {
  	button.disabled = false
  }
})
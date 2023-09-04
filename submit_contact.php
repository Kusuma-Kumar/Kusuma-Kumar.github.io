<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $full_name = $_POST["full-name"];
    $email = $_POST["email"];
    $message = $_POST["message"];

    // Define the recipient email address
    $to = "kusumakumar2003@gmail.com";

    // Define the subject and message
    $subject = "Contact Form Submission from $full_name";
    $message_body = "Full Name: $full_name\n";
    $message_body .= "Email: $email\n";
    $message_body .= "Message:\n$message";

    // Additional headers
    $headers = "From: $email";

    // Send the email
    $success = mail($to, $subject, $message_body, $headers);

    if ($success) {
        echo "<h2>Thank you, $full_name!</h2>";
        echo "<p>Your message has been sent successfully.</p>";
    } else {
        echo "<h2>Oops, something went wrong!</h2>";
        echo "<p>Sorry, there was an issue sending your message. Please try again later.</p>";
    }
} else {
    header("Location: index.html"); // Redirect back to the contact form if accessed directly.
}
error_reporting(E_ALL);
ini_set('display_errors', 1);
?>

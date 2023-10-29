import { Fragment, useContext, useEffect, useRef, useState } from "react"
import { useRouter } from "next/router"
import { Event, getAllLocalStorageItems, getRefValue, getRefValues, isTrue, preventDefault, refs, set_val, spreadArraysOrObjects, uploadFiles, useEventLoop } from "/utils/state"
import { ColorModeContext, EventLoopContext, initialEvents, StateContext } from "/utils/context.js"
import "focus-visible/dist/focus-visible"
import { Box, Button, FormControl, FormLabel, Heading, Input, Link, Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay, Text } from "@chakra-ui/react"
import { getEventURL } from "/utils/state.js"
import NextLink from "next/link"
import NextHead from "next/head"



export default function Component() {
  const state = useContext(StateContext)
  const router = useRouter()
  const [ colorMode, toggleColorMode ] = useContext(ColorModeContext)
  const focusRef = useRef();
  
  // Main event loop.
  const [addEvents, connectError] = useContext(EventLoopContext)

  // Set focus to the specified element.
  useEffect(() => {
    if (focusRef.current) {
      focusRef.current.focus();
    }
  })

  // Route after the initial page hydration.
  useEffect(() => {
    const change_complete = () => addEvents(initialEvents())
    router.events.on('routeChangeComplete', change_complete)
    return () => {
      router.events.off('routeChangeComplete', change_complete)
    }
  }, [router])

  const ref_username = useRef(null); refs['ref_username'] = ref_username;
  const ref_password = useRef(null); refs['ref_password'] = ref_password;

  return (
    <Fragment>
  <Fragment>
  {isTrue(connectError !== null) ? (
  <Fragment>
  <Modal isOpen={connectError !== null}>
  <ModalOverlay>
  <ModalContent>
  <ModalHeader>
  {`Connection Error`}
</ModalHeader>
  <ModalBody>
  <Text>
  {`Cannot connect to server: `}
  {(connectError !== null) ? connectError.message : ''}
  {`. Check if server is reachable at `}
  {getEventURL().href}
</Text>
</ModalBody>
</ModalContent>
</ModalOverlay>
</Modal>
</Fragment>
) : (
  <Fragment/>
)}
</Fragment>
  <Fragment>
  <Box className={`flex min-h-full flex-col justify-center px-6 py-12 lg:px-8`}>
  <Box className={`sm:mx-auto sm:w-full sm:max-w-sm`}>
  <Heading className={`mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900`} size={`md`}>
  {`Sign in to your account`}
</Heading>
</Box>
  <Fragment>
  {isTrue(state.is_hydrated) ? (
  <Fragment>
  <Box className={`mt-10 sm:mx-auto sm:w-full sm:max-w-sm`}>
  <Fragment>
  {isTrue((state.login_state.error_message !== "")) ? (
  <Fragment>
  <Text>
  {state.login_state.error_message}
</Text>
</Fragment>
) : (
  <Fragment/>
)}
</Fragment>
  <Box as={`form`} className={`space-y-6`} onSubmit={(_e0) => addEvents([Event("state.login_state.on_submit", {form_data:{"password": getRefValue(ref_password), "username": getRefValue(ref_username)}})], (_e0))}>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`username`}>
  {`Username`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`username`} ref={ref_username} type={`text`}/>
</Box>
</FormControl>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`password`}>
  {`Password`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`password`} ref={ref_password} type={`password`}/>
</Box>
</FormControl>
  <Button className={`flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`} type={`submit`}>
  {`Sign in`}
</Button>
</Box>
  <Text className={`mt-10 text-center text-sm text-gray-500`}>
  {`Don't have an account? `}
  <Link as={NextLink} className={`font-semibold leading-6 text-indigo-600 hover:text-indigo-500`} href={`/register`}>
  {`Sign up`}
</Link>
</Text>
</Box>
</Fragment>
) : (
  <Fragment/>
)}
</Fragment>
</Box>
</Fragment>
  <NextHead>
  <title>
  {`Reflex App`}
</title>
  <meta content={`A Reflex app.`} name={`description`}/>
  <meta content={`favicon.ico`} property={`og:image`}/>
</NextHead>
</Fragment>
  )
}

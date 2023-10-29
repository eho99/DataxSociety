import { Fragment, useContext, useEffect, useRef, useState } from "react"
import { useRouter } from "next/router"
import { Event, getAllLocalStorageItems, getRefValue, getRefValues, isTrue, preventDefault, refs, set_val, spreadArraysOrObjects, uploadFiles, useEventLoop } from "/utils/state"
import { ColorModeContext, EventLoopContext, initialEvents, StateContext } from "/utils/context.js"
import "focus-visible/dist/focus-visible"
import { Box, Button, FormControl, FormLabel, Heading, Input, Modal, ModalBody, ModalContent, ModalHeader, ModalOverlay, Text, Textarea } from "@chakra-ui/react"
import { getEventURL } from "/utils/state.js"
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

  const ref_description = useRef(null); refs['ref_description'] = ref_description;
  const ref_project_name = useRef(null); refs['ref_project_name'] = ref_project_name;
  const ref_num_data = useRef(null); refs['ref_num_data'] = ref_num_data;
  const ref_num_indep_vars = useRef(null); refs['ref_num_indep_vars'] = ref_num_indep_vars;
  const ref_num_dep_vars = useRef(null); refs['ref_num_dep_vars'] = ref_num_dep_vars;

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
  {isTrue(state.is_hydrated) ? (
  <Fragment>
  <Box>
  <Box>
  <Box dangerouslySetInnerHTML={{"__html": "\n        <nav class=\"bg-white border-gray-200 dark:bg-gray-900 dark:border-gray-700\">\n            <div class=\"max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4\">\n                <a href=\"#\" class=\"flex items-center\">\n                    <span class=\"self-center text-2xl font-semibold whitespace-nowrap dark:text-white\">DataxSociety</span>\n                </a>\n                <ul class=\"flex flex-col font-medium p-4 md:p-0 mt-4 border border-gray-100 rounded-lg bg-gray-50 md:flex-row md:space-x-8 md:mt-0 md:border-0 md:bg-white dark:bg-gray-800 md:dark:bg-gray-900 dark:border-gray-700\">                \n                    <li>\n                    <a href=\"/\" class=\"block py-2 pl-3 pr-4 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent\">Home</a>\n                    </li>\n                    <li>\n                    <a href=\"/create_project\" class=\"block py-2 pl-3 pr-4 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent\">Create new project</a>\n                    </li>\n                    <li>\n                    <a href=\"/\" class=\"block py-2 pl-3 pr-4 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent\">My projects</a>\n                    </li>\n                    <li>\n                    <a href=\"/login\" class=\"block py-2 pl-3 pr-4 text-gray-900 rounded hover:bg-gray-100 md:hover:bg-transparent md:border-0 md:hover:text-blue-700 md:p-0 dark:text-white md:dark:hover:text-blue-500 dark:hover:bg-gray-700 dark:hover:text-white md:dark:hover:bg-transparent\">Sign out</a>\n                    </li>\n                </ul>\n                </div>\n            </div>\n        </nav>\n        "}}/>
</Box>
  <Heading>
  {`Create a new project`}
</Heading>
  <Box as={`form`} className={`space-y-6`} onSubmit={(_e0) => addEvents([Event("state.create_project_state.on_submit", {form_data:{"num_indep_vars": getRefValue(ref_num_indep_vars), "description": getRefValue(ref_description), "project_name": getRefValue(ref_project_name), "num_dep_vars": getRefValue(ref_num_dep_vars), "num_data": getRefValue(ref_num_data)}})], (_e0))}>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`project_name`}>
  {`Project Name`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`project_name`} ref={ref_project_name} type={`text`}/>
</Box>
</FormControl>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`description`}>
  {`Description`}
</FormLabel>
  <Box className={`mt-2`}>
  <Textarea className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`description`} ref={ref_description}/>
</Box>
</FormControl>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`num_indep_vars`}>
  {`Number of Independent Variables`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`num_indep_vars`} ref={ref_num_indep_vars} sx={{"type": "number"}} type={`text`}/>
</Box>
</FormControl>
  <FormControl isRequired={true}>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`num_dep_vars`}>
  {`Number of Dependent Variables`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`num_dep_vars`} ref={ref_num_dep_vars} sx={{"type": "number"}} type={`text`}/>
</Box>
</FormControl>
  <FormControl>
  <FormLabel className={`block text-sm font-medium leading-6 text-gray-900`} htmlFor={`num_data`}>
  {`Number of data points`}
</FormLabel>
  <Box className={`mt-2`}>
  <Input className={`block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6`} id={`num_data`} ref={ref_num_data} sx={{"type": "number"}} type={`text`}/>
</Box>
</FormControl>
  <Button className={`flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-black shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`} type={`submit`}>
  {`Create Project`}
</Button>
</Box>
</Box>
</Fragment>
) : (
  <Fragment/>
)}
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
